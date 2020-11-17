from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON

handler_1 = "Experiments/results/bAbI two supporting facts experiments.json"
handler_2 = "Experiments/results/model_bAbI two_supporting_facts_experiments.json"


handler_1 = ApplicationHandlerFileHandlerJSON(handler_1)
handler_2 = ApplicationHandlerFileHandlerJSON(handler_2)

all_handlers = handler_1.read_application_handlers_from_file() + handler_2.read_application_handlers_from_file()

handler_1.write_application_handlers_to_file(all_handlers)