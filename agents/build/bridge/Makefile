CXX = g++

REPO_ROOT = ../../..
AGENTS_ROOT = $(REPO_ROOT)/agents

CXXFLAGS = -std=c++17 -O3 -ggdb -march=native
CXXFLAGS += -Wall -Wextra -Wpedantic -Wno-implicit-fallthrough -Wno-unused-parameter -Wno-class-memaccess
CXXFLAGS += -I$(REPO_ROOT)/third_party/jsoncpp/include
CXXFLAGS += -I$(REPO_ROOT)/third_party/tiny-dnn
CXXFLAGS += -I$(REPO_ROOT)/engine/include
CXXFLAGS += -I$(REPO_ROOT)/judge/include
CXXFLAGS += -I$(AGENTS_ROOT)/include
CXXFLAGS += -I$(AGENTS_ROOT)/test

LDFLAGS = -lpthread -O3

OBJDIR = obj

JSON_VALUE_SRC = $(REPO_ROOT)/third_party/jsoncpp/src/json_value.cpp
JSON_READER_SRC = $(REPO_ROOT)/third_party/jsoncpp/src/json_reader.cpp
JSON_WRITER_SRC = $(REPO_ROOT)/third_party/jsoncpp/src/json_writer.cpp
NN_SRC = $(AGENTS_ROOT)/src/neural_net/NeuralNetwork.cpp
DISPATCHER_SRC = $(AGENTS_ROOT)/test/CardDispatcher.cpp
STATE_BUILDER_SRC = $(AGENTS_ROOT)/test/TestStateBuilder.cpp
MAIN_SRC = $(AGENTS_ROOT)/src/bridge/bridge_main.cpp

OBJS = \
	$(OBJDIR)/json_value.o \
	$(OBJDIR)/json_reader.o \
	$(OBJDIR)/json_writer.o \
	$(OBJDIR)/NeuralNetwork.o \
	$(OBJDIR)/CardDispatcher.o \
	$(OBJDIR)/TestStateBuilder.o \
	$(OBJDIR)/bridge_main.o

CARDS_JSON = cards.json
CARDS_JSON_SRC = $(REPO_ROOT)/engine/include/Cards/cards.json

EXE = hearthstone_bridge

.PHONY: all clean

all: $(EXE) $(CARDS_JSON)
	@echo "Done."

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(CARDS_JSON): $(CARDS_JSON_SRC)
	cp $(CARDS_JSON_SRC) $(CARDS_JSON)

$(OBJDIR)/json_value.o: $(JSON_VALUE_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/json_reader.o: $(JSON_READER_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/json_writer.o: $(JSON_WRITER_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/NeuralNetwork.o: $(NN_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/CardDispatcher.o: $(DISPATCHER_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/TestStateBuilder.o: $(STATE_BUILDER_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/bridge_main.o: $(MAIN_SRC) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXE): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf $(OBJDIR) $(EXE) $(CARDS_JSON)
