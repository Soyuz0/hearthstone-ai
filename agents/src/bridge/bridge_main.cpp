#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include <json/json.h>

#include "Cards/Database.h"
#include "Cards/PreIndexedCards.h"
#include "MCTS/inspector/InteractiveShell.h"
#include "agents/MCTSConfig.h"
#include "agents/MCTSRunner.h"
#include "decks/Decks.h"
#include "engine/ActionApplyHelper.h"
#include "engine/Game-impl.h"
#include "engine/MainOp.h"
#include "engine/ValidActionAnalyzer.h"
#include "engine/view/BoardRefView.h"
#include "engine/view/BoardView.h"
#include "engine/view/board_view/StateRestorer.h"
#include "neural_net/NeuralNetwork.h"

namespace {

constexpr double kDefaultThinkSeconds = 1.0;
constexpr int kDefaultThreads = 1;
constexpr int kDefaultTreeSamples = 64;
constexpr const char* kDefaultDeckType = "InnKeeperExpertWarlock";
constexpr const char* kDefaultNeuralNetPath = "neural_net_bridge";
constexpr bool kDefaultTreatNeuralNetAsRandom = true;

bool StartsWith(std::string const& value, std::string const& prefix) {
	if (value.size() < prefix.size()) return false;
	return std::equal(prefix.begin(), prefix.end(), value.begin());
}

bool EnvFlagEnabled(char const* name) {
	auto const* raw = std::getenv(name);
	if (!raw) return false;
	std::string value(raw);
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string MainOpToChoiceType(engine::MainOpType op) {
	switch (op) {
	case engine::kMainOpPlayCard:
		return "PlayCard";
	case engine::kMainOpAttack:
		return "Attack";
	case engine::kMainOpHeroPower:
		return "HeroPower";
	case engine::kMainOpEndTurn:
		return "EndTurn";
	default:
		return "Unknown";
	}
}

std::string DescribeEncodedTarget(int idx) {
	std::string side = "Your";
	if (idx >= 8) {
		side = "Opponent";
		idx -= 8;
	}
	if (idx == 7) return side + " Hero";
	std::stringstream ss;
	ss << side << " " << (idx + 1) << "th Minion";
	return ss.str();
}

int EncodeTarget(state::State const& board, state::CardRef ref) {
	if (!ref.IsValid()) return -1;

	auto const& first = board.GetBoard().GetFirst();
	if (first.GetHeroRef() == ref) return 7;
	int first_idx = 0;
	bool first_found = false;
	first.minions_.ForEach([&](state::CardRef card_ref) {
		if (card_ref == ref) {
			first_found = true;
			return false;
		}
		++first_idx;
		return true;
	});
	if (first_found) return first_idx;

	auto const& second = board.GetBoard().GetSecond();
	if (second.GetHeroRef() == ref) return 15;
	int second_idx = 0;
	bool second_found = false;
	second.minions_.ForEach([&](state::CardRef card_ref) {
		if (card_ref == ref) {
			second_found = true;
			return false;
		}
		++second_idx;
		return true;
	});
	if (second_found) return second_idx + 8;

	return -1;
}

Json::Value MakeError(std::string const& error) {
	Json::Value out(Json::objectValue);
	out["error"] = error;
	return out;
}

std::string CompactJson(Json::Value const& value) {
	Json::StreamWriterBuilder builder;
	builder["indentation"] = "";
	builder["commentStyle"] = "None";
	return Json::writeString(builder, value);
}

void WriteJsonLine(Json::Value const& value) {
	std::cout << CompactJson(value) << std::endl;
	std::cout.flush();
}

bool EnsureNeuralNetFile(std::string const& path, bool& created_file) {
	created_file = false;
	std::ifstream in(path);
	if (in.good()) return true;
	try {
		neural_net::NeuralNetwork::CreateWithRandomWeights(path);
		created_file = true;
		return true;
	}
	catch (...) {
		return false;
	}
}

std::unordered_map<int, std::string> BuildReverseCardMap() {
	std::unordered_map<int, std::string> reverse;
	auto const& id_map = Cards::Database::GetInstance().GetIdMap();
	for (auto const& item : id_map) {
		reverse[item.second] = item.first;
	}
	return reverse;
}

std::string CardIdIntToString(
	int card_id,
	std::unordered_map<int, std::string> const& reverse_card_map) {
	if (card_id <= 0) return "";
	auto it = reverse_card_map.find(card_id);
	if (it == reverse_card_map.end()) return "";
	return it->second;
}

std::string DefaultHeroPowerCardIdForHero(int hero_card_id) {
	if (hero_card_id <= 0) return "";

	auto const& hero = Cards::Database::GetInstance().Get(hero_card_id);
	switch (hero.player_class) {
	case state::kPlayerClassDruid:
		return "CS2_017";
	case state::kPlayerClassHunter:
		return "DS1h_292";
	case state::kPlayerClassMage:
		return "CS2_034";
	case state::kPlayerClassPaladin:
		return "CS2_101";
	case state::kPlayerClassPriest:
		return "CS1h_001";
	case state::kPlayerClassRogue:
		return "CS2_083";
	case state::kPlayerClassShaman:
		return "CS2_049";
	case state::kPlayerClassWarlock:
		return "CS2_056";
	case state::kPlayerClassWarrior:
		return "CS2_102";
	default:
		return "";
	}
}

bool IsLikelyBoardViewFormat(Json::Value const& board_json) {
	if (!board_json.isObject()) return false;
	if (!board_json.isMember("entities")) return false;
	if (!board_json.isMember("player")) return false;
	if (!board_json.isMember("opponent")) return false;
	if (!board_json["player"].isObject()) return false;
	if (!board_json["player"].isMember("hero")) return false;
	if (!board_json["player"]["hero"].isObject()) return false;
	return board_json["player"]["hero"].isMember("damage");
}

int AddEntity(std::string const& card_id, Json::Value& entities, int& next_entity_id) {
	Json::Value entity(Json::objectValue);
	entity["card_id"] = card_id;
	entity["generate_under_blocks"] = Json::Value(Json::arrayValue);

	int entity_id = next_entity_id;
	++next_entity_id;

	if (entities.size() <= static_cast<Json::ArrayIndex>(entity_id)) {
		entities.resize(entity_id + 1);
	}
	entities[entity_id] = entity;
	return entity_id;
}

bool ConvertCompactPlayer(
	Json::Value const& compact,
	Json::Value& player_out,
	Json::Value& entities,
	int& next_entity_id,
	std::unordered_map<int, std::string> const& reverse_card_map,
	std::string& error) {
	Json::Value hero = compact["hero"];
	Json::Value hero_power = compact["hero_power"];
	if (!hero.isObject() || !hero_power.isObject()) {
		error = "Compact board is missing hero or hero_power";
		return false;
	}

	int hero_card_id = hero.get("card_id", 0).asInt();
	std::string hero_card_str = CardIdIntToString(hero_card_id, reverse_card_map);
	if (hero_card_str.empty()) {
		error = "Unknown hero card_id in compact board";
		return false;
	}

	int hero_power_id = hero_power.get("card_id", 0).asInt();
	std::string hero_power_card_str = CardIdIntToString(hero_power_id, reverse_card_map);
	if (hero_power_card_str.empty()) {
		hero_power_card_str = DefaultHeroPowerCardIdForHero(hero_card_id);
	}
	if (hero_power_card_str.empty()) {
		error = "Unknown hero power card_id in compact board";
		return false;
	}

	int hero_hp = hero.get("hp", hero.get("health", 30)).asInt();
	int hero_max_hp = hero.get("max_hp", 30).asInt();
	int hero_attack = hero.get("attack", 0).asInt();
	bool hero_attackable = hero.get("attackable", false).asBool();

	Json::Value hero_json(Json::objectValue);
	hero_json["card_id"] = hero_card_str;
	hero_json["max_hp"] = hero_max_hp;
	hero_json["damage"] = std::max(0, hero_max_hp - hero_hp);
	hero_json["armor"] = hero.get("armor", 0).asInt();
	hero_json["attack"] = hero_attack;
	hero_json["attacks_this_turn"] = hero_attackable ? 0 : ((hero_attack > 0) ? 1 : 0);

	Json::Value hero_status(Json::objectValue);
	hero_status["charge"] = hero.get("charge", false).asBool();
	hero_status["taunt"] = hero.get("taunt", false).asBool();
	hero_status["divine_shield"] = hero.get("divine_shield", false).asBool();
	hero_status["stealth"] = hero.get("stealth", false).asBool();
	hero_status["freeze"] = hero.get("freeze", false).asBool();
	hero_status["frozon"] = hero.get("frozen", false).asBool();
	hero_status["poisonous"] = hero.get("poisonous", false).asBool();
	hero_status["windfury"] = hero.get("windfury", false).asBool();
	hero_json["status"] = hero_status;

	Json::Value hero_power_json(Json::objectValue);
	hero_power_json["card_id"] = hero_power_card_str;
	hero_power_json["used"] = hero_power.get("used", false).asBool();
	hero_json["hero_power"] = hero_power_json;

	player_out["hero"] = hero_json;

	Json::Value minions_out(Json::arrayValue);
	Json::Value minions_in = compact["minions"];
	if (minions_in.isArray()) {
		for (Json::ArrayIndex i = 0; i < minions_in.size(); ++i) {
			Json::Value const& minion = minions_in[i];
			int minion_card_id = minion.get("card_id", 0).asInt();
			std::string minion_card_str = CardIdIntToString(minion_card_id, reverse_card_map);
			if (minion_card_str.empty()) {
				error = "Unknown minion card_id in compact board";
				return false;
			}

			int minion_hp = minion.get("hp", minion.get("health", 1)).asInt();
			int minion_max_hp = minion.get("max_hp", std::max(1, minion_hp)).asInt();
			int minion_attack = minion.get("attack", 0).asInt();
			bool minion_attackable = minion.get("attackable", false).asBool();

			Json::Value minion_out(Json::objectValue);
			minion_out["card_id"] = minion_card_str;
			minion_out["max_hp"] = minion_max_hp;
			minion_out["damage"] = std::max(0, minion_max_hp - minion_hp);
			minion_out["attack"] = minion_attack;
			minion_out["attacks_this_turn"] = minion_attackable ? 0 : ((minion_attack > 0) ? 1 : 0);

			Json::Value minion_status(Json::objectValue);
			minion_status["charge"] = minion.get("charge", false).asBool();
			minion_status["taunt"] = minion.get("taunt", false).asBool();
			minion_status["divine_shield"] = minion.get("divine_shield", false).asBool();
			minion_status["stealth"] = minion.get("stealth", false).asBool();
			minion_status["freeze"] = minion.get("freeze", false).asBool();
			minion_status["frozon"] = minion.get("frozen", false).asBool();
			minion_status["poisonous"] = minion.get("poisonous", false).asBool();
			minion_status["windfury"] = minion.get("windfury", false).asBool();
			minion_out["status"] = minion_status;

			minion_out["silenced"] = minion.get("silenced", false).asBool();
			minion_out["spellpower"] = minion.get("spellpower", 0).asInt();
			minion_out["summoned_this_turn"] = minion.get(
				"summoned_this_turn",
				(!minion_attackable && !minion.get("charge", false).asBool())).asBool();
			minion_out["exhausted"] = !minion_attackable;

			minions_out.append(minion_out);
		}
	}
	player_out["minions"] = minions_out;

	player_out["fatigue"] = compact.get("fatigue", 0).asInt();

	Json::Value resource = compact["resource"];
	int current = resource.get("current", 0).asInt();
	int total = resource.get("total", 0).asInt();
	int this_turn = std::max(0, current - total);
	int used = std::max(0, total - current);
	Json::Value crystal(Json::objectValue);
	crystal["this_turn"] = this_turn;
	crystal["used"] = used;
	crystal["total"] = total;
	crystal["overload"] = resource.get("overload", 0).asInt();
	crystal["overload_next_turn"] = resource.get("overload_next", 0).asInt();
	player_out["crystal"] = crystal;

	Json::Value deck_out(Json::objectValue);
	Json::Value deck_entities(Json::arrayValue);
	int deck_count = compact.get("deck_count", 0).asInt();
	for (int i = 0; i < deck_count; ++i) {
		int entity_id = AddEntity("", entities, next_entity_id);
		deck_entities.append(entity_id);
	}
	deck_out["entities"] = deck_entities;
	player_out["deck"] = deck_out;

	Json::Value hand_out(Json::objectValue);
	Json::Value hand_entities(Json::arrayValue);
	Json::Value hand = compact["hand"];
	if (hand.isArray()) {
		for (Json::ArrayIndex i = 0; i < hand.size(); ++i) {
			int hand_card_id = hand[i].get("card_id", 0).asInt();
			std::string hand_card_str = CardIdIntToString(hand_card_id, reverse_card_map);
			int entity_id = AddEntity(hand_card_str, entities, next_entity_id);
			hand_entities.append(entity_id);
		}
	}
	hand_out["entities"] = hand_entities;
	player_out["hand"] = hand_out;

	return true;
}

bool ConvertCompactBoard(
	Json::Value const& compact,
	Json::Value& converted,
	std::unordered_map<int, std::string> const& reverse_card_map,
	std::string& error) {
	if (!compact.isObject()) {
		error = "BOARD payload must be a JSON object";
		return false;
	}

	if (!compact.isMember("player") || !compact.isMember("opponent")) {
		error = "Compact board is missing player/opponent";
		return false;
	}

	converted = Json::Value(Json::objectValue);
	converted["turn"] = compact.get("turn", 1).asInt();

	Json::Value entities(Json::arrayValue);
	int next_entity_id = 1;

	Json::Value player_out(Json::objectValue);
	if (!ConvertCompactPlayer(
		compact["player"],
		player_out,
		entities,
		next_entity_id,
		reverse_card_map,
		error)) {
		return false;
	}

	Json::Value opponent_out(Json::objectValue);
	if (!ConvertCompactPlayer(
		compact["opponent"],
		opponent_out,
		entities,
		next_entity_id,
		reverse_card_map,
		error)) {
		return false;
	}

	converted["player"] = player_out;
	converted["opponent"] = opponent_out;
	converted["entities"] = entities;
	return true;
}

std::optional<int> GetBestChoice(mcts::selection::TreeNode const* node) {
	if (!node) return std::nullopt;

	int best_choice = -1;
	int64_t best_chosen_times = -1;
	node->children_.ForEach([&](int choice, mcts::selection::EdgeAddon const* edge_addon, mcts::selection::TreeNode* child) {
		(void)child;
		if (!edge_addon) return true;
		auto chosen_times = edge_addon->GetChosenTimes();
		if (chosen_times > best_chosen_times) {
			best_chosen_times = chosen_times;
			best_choice = choice;
		}
		return true;
	});

	if (best_choice < 0) return std::nullopt;
	if (best_chosen_times <= 0) return std::nullopt;
	return best_choice;
}

Json::Value ExtractBestActions(
	agents::MCTSRunner& controller,
	state::State const& start_state,
	engine::view::board_view::StateRestorer& state_restorer) {
	auto const root_side = start_state.GetCurrentPlayerId().GetSide();
	auto const* root = controller.GetRootNode(root_side);
	if (!root) return MakeError("MCTS root node unavailable");

	if (!root->addon_.consistency_checker.CheckActionType(engine::ActionType::kMainAction)) {
		return MakeError("MCTS root node has unexpected action type");
	}

	engine::ActionApplyHelper action_helper;
	Json::Value actions(Json::arrayValue);

	auto const* node = root;
	for (int depth = 0; depth < 64 && node; ++depth) {
		auto best_choice = GetBestChoice(node);
		if (!best_choice.has_value()) break;
		int choice = *best_choice;

		std::mt19937 traversal_rand(0);
		state::State traversed_state = state_restorer.RestoreState(traversal_rand);
		engine::Result traversal_result = engine::kResultInvalid;
		action_helper.ApplyChoices(traversed_state, traversal_result);

		engine::ValidActionAnalyzer action_analyzer;
		action_analyzer.Analyze(traversed_state);

		Json::Value action(Json::objectValue);
		auto action_type = node->addon_.consistency_checker.GetActionType().GetType();

		if (action_type == engine::ActionType::kMainAction) {
			action["type"] = "main_action";
			action["choice"] = choice;

			auto main_actions_count = action_analyzer.GetMainActionsCount();
			if (choice >= 0 && choice < main_actions_count) {
				action["choice_type"] = MainOpToChoiceType(action_analyzer.GetMainOpType(choice));
			}
			else {
				action["choice_type"] = "Unknown";
			}
		}
		else if (action_type == engine::ActionType::kChooseHandCard) {
			action["type"] = "choose_hand_card";
			action["choice"] = choice;

			auto const& playable = action_analyzer.GetPlayableCards();
			if (choice >= 0 && choice < static_cast<int>(playable.size())) {
				auto hand_index = static_cast<int>(playable[choice]);
				action["hand_index"] = hand_index;
				if (hand_index >= 0 && hand_index < static_cast<int>(traversed_state.GetCurrentPlayer().hand_.Size())) {
					auto card_ref = traversed_state.GetCurrentPlayer().hand_.Get(static_cast<size_t>(hand_index));
					auto card_id = static_cast<int>(traversed_state.GetCard(card_ref).GetCardId());
					action["card_id"] = card_id;
					if (card_id > 0) {
						action["card_name"] = Cards::Database::GetInstance().Get(card_id).name;
					}
				}
			}
		}
		else if (action_type == engine::ActionType::kChooseAttacker) {
			action["type"] = "choose_attacker";
			action["choice"] = choice;

			auto const& attackers = action_analyzer.GetAttackers();
			if (choice >= 0 && choice < static_cast<int>(attackers.size())) {
				action["encoded_attacker"] = attackers[choice];
			}
		}
		else if (action_type == engine::ActionType::kChooseDefender) {
			action["type"] = "choose_target";
			action["choice"] = choice;

			engine::ActionApplyHelper probe_helper = action_helper;
			probe_helper.AppendChoice(choice);
			std::mt19937 probe_rand(0);
			state::State probe_state = state_restorer.RestoreState(probe_rand);
			auto callback_info = probe_helper.ApplyChoices(probe_state);
			if (std::holds_alternative<engine::ActionApplyHelper::ChooseDefenderInfo>(callback_info)) {
				auto const& info = std::get<engine::ActionApplyHelper::ChooseDefenderInfo>(callback_info);
				if (choice >= 0 && choice < static_cast<int>(info.targets.size())) {
					int encoded = info.targets[choice];
					action["encoded_target"] = encoded;
					action["target_desc"] = DescribeEncodedTarget(encoded);
				}
			}
		}
		else if (action_type == engine::ActionType::kChooseTarget) {
			action["type"] = "choose_target";
			action["choice"] = choice;

			engine::ActionApplyHelper probe_helper = action_helper;
			probe_helper.AppendChoice(choice);
			std::mt19937 probe_rand(0);
			state::State probe_state = state_restorer.RestoreState(probe_rand);
			auto callback_info = probe_helper.ApplyChoices(probe_state);
			if (std::holds_alternative<engine::ActionApplyHelper::GetSpecifiedTargetInfo>(callback_info)) {
				auto const& info = std::get<engine::ActionApplyHelper::GetSpecifiedTargetInfo>(callback_info);
				if (choice >= 0 && choice < static_cast<int>(info.targets.size())) {
					int encoded = EncodeTarget(probe_state, info.targets[choice]);
					if (encoded >= 0) {
						action["encoded_target"] = encoded;
						action["target_desc"] = DescribeEncodedTarget(encoded);
					}
				}
			}
		}
		else if (action_type == engine::ActionType::kChooseOne) {
			action["type"] = "choose_one";
			action["choice"] = choice;
		}
		else if (action_type == engine::ActionType::kChooseMinionPutLocation) {
			action["type"] = "choose_minion_put_location";
			action["choice"] = choice;
		}
		else {
			action["type"] = "choice";
			action["choice"] = choice;
		}

		actions.append(action);
		action_helper.AppendChoice(choice);

		auto child_info = node->children_.Get(choice);
		node = child_info.second;
	}

	if (actions.empty()) {
		return MakeError("MCTS did not produce a best action path");
	}

	Json::Value end_action(Json::objectValue);
	end_action["type"] = "end_action";
	actions.append(end_action);

	Json::Value out(Json::objectValue);
	out["actions"] = actions;
	return out;
}

Json::Value ProcessBoard(
	std::string const& board_payload,
	double think_time_sec,
	agents::MCTSAgentConfig const& config,
	std::unordered_map<int, std::string> const& reverse_card_map) {
	Json::CharReaderBuilder builder;
	std::string errs;
	Json::Value input_board;
	{
		std::istringstream iss(board_payload);
		if (!Json::parseFromStream(builder, iss, &input_board, &errs)) {
			return MakeError("Failed to parse BOARD JSON: " + errs);
		}
	}

	Json::Value board_for_parser;
	if (IsLikelyBoardViewFormat(input_board)) {
		board_for_parser = input_board;
	}
	else {
		std::string convert_error;
		if (!ConvertCompactBoard(input_board, board_for_parser, reverse_card_map, convert_error)) {
			return MakeError("Failed to convert compact board: " + convert_error);
		}
	}

	engine::view::BoardView board_view;
	board_view.Reset();

	engine::view::board_view::UnknownCardsInfo first_unknown;
	engine::view::board_view::UnknownCardsInfo second_unknown;
	first_unknown.deck_cards_ = decks::Decks::GetDeckCards(kDefaultDeckType);
	second_unknown.deck_cards_ = decks::Decks::GetDeckCards(kDefaultDeckType);

	try {
		board_view.Parse(board_for_parser, first_unknown, second_unknown);
	}
	catch (std::exception const& ex) {
		return MakeError(std::string("BoardView::Parse failed: ") + ex.what());
	}
	catch (...) {
		return MakeError("BoardView::Parse failed with unknown error");
	}

	auto state_restorer = engine::view::board_view::StateRestorer::Prepare(
		board_view,
		first_unknown,
		second_unknown);

	std::random_device seed;
	std::mt19937 rand(seed());
	state::State start_state = state_restorer.RestoreState(rand);

	agents::MCTSRunner controller(config, rand);
	controller.Run(engine::view::BoardRefView(start_state, start_state.GetCurrentPlayerId().GetSide()));

	double clamped_time = std::max(0.05, think_time_sec);
	auto run_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(
		static_cast<int>(clamped_time * 1000.0));
	while (std::chrono::steady_clock::now() < run_until) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}

	controller.NotifyStop();
	controller.WaitUntilStopped();

	return ExtractBestActions(controller, start_state, state_restorer);
}

bool InitializeBridge(
	agents::MCTSAgentConfig& config,
	std::unordered_map<int, std::string>& reverse_card_map,
	std::string& error) {
	if (!Cards::Database::GetInstance().Initialize("cards.json")) {
		error = "Failed to initialize card database from cards.json";
		return false;
	}
	Cards::PreIndexedCards::GetInstance().Initialize();

	reverse_card_map = BuildReverseCardMap();

	bool created_neural_net_file = false;
	if (!EnsureNeuralNetFile(kDefaultNeuralNetPath, created_neural_net_file)) {
		error = "Failed to create/load neural network file";
		return false;
	}

	bool disable_net_bias = kDefaultTreatNeuralNetAsRandom;
	if (EnvFlagEnabled("PETER_BRIDGE_USE_TRAINED_NET")) {
		disable_net_bias = false;
	}
	if (created_neural_net_file) {
		disable_net_bias = true;
	}

	config.threads = kDefaultThreads;
	config.tree_samples = kDefaultTreeSamples;
	config.mcts.SetNeuralNetPath(kDefaultNeuralNetPath, disable_net_bias);
	return true;
}

} // namespace

int main() {
	agents::MCTSAgentConfig config;
	std::unordered_map<int, std::string> reverse_card_map;
	std::string init_error;
	if (!InitializeBridge(config, reverse_card_map, init_error)) {
		WriteJsonLine(MakeError(init_error));
		return 1;
	}

	double think_time_sec = kDefaultThinkSeconds;

	std::string line;
	while (std::getline(std::cin, line)) {
		if (line == "QUIT") {
			return 0;
		}

		if (StartsWith(line, "THINK_TIME:")) {
			std::string value = line.substr(std::string("THINK_TIME:").size());
			std::stringstream ss(value);
			double parsed = 0.0;
			if (!(ss >> parsed) || parsed <= 0.0) {
				WriteJsonLine(MakeError("Invalid THINK_TIME value"));
				continue;
			}
			think_time_sec = parsed;
			continue;
		}

		if (StartsWith(line, "BOARD:")) {
			std::string board_payload = line.substr(std::string("BOARD:").size());
			try {
				Json::Value response = ProcessBoard(board_payload, think_time_sec, config, reverse_card_map);
				WriteJsonLine(response);
			}
			catch (std::exception const& ex) {
				WriteJsonLine(MakeError(std::string("Bridge execution failed: ") + ex.what()));
			}
			catch (...) {
				WriteJsonLine(MakeError("Bridge execution failed with unknown error"));
			}
			continue;
		}

		WriteJsonLine(MakeError("Unknown command"));
	}

	return 0;
}
