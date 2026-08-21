.PHONY: test demo gif clean

PY ?= python3

test:
	$(PY) -m unittest discover -s tests -v

demo:
	$(PY) -m agentlab demo

gif: demo
	bash demo/make_demo.sh

clean:
	rm -rf demo/session-todo.json demo/session-memory.jsonl demo/session.txt demo/frames