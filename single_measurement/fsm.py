from automat import MethodicalMachine
from utils import construct_measurement_dictionary, get_band, int2onehot

class Controller():
    _machine = MethodicalMachine()

    def __init__(self):
        self.neighbor_stack = []
        self.state = "clear_bandlock"
        self.adjust_search_params_not_ok = False
        self.last_adjust_search_params_command = ""
        self.measurement_result_batch = []

    @_machine.state()
    def activate(self):
        "In this state, modem activation command is sent."

    @_machine.state()
    def measure(self):
        "In this state, modem sends measure command"

    @_machine.state()
    def wait_measurement_result(self):
        "In this state, modem sends measure command"



    @_machine.state(initial=True)
    def clear_bandlock(self):
        ""

    @_machine.input()
    def ok(self, response=None):
        "ok command received in activate state"

    @_machine.input()
    def not_ok(self):
        "ok command not received in activate state"


    @_machine.input()
    def get_next_command(self):
        "return next command"


    @_machine.output()
    def get_activate_command(self):
        return "AT+CFUN=1"


    @_machine.output()
    def get_measure_command(self):
        return "AT%NCELLMEAS"

    @_machine.output()
    def get_empty_command(self):
        return ""


    @_machine.output()
    def save_measurement(self, response):
        if response is None:
            return
        current_measurement_dictionary = construct_measurement_dictionary(response)
        print(current_measurement_dictionary)
        measurement_result_batch_to_return = self.measurement_result_batch.copy()
        self.measurement_result_batch.clear()
        self.measurement_result_batch.append(response)

        if "neighbor_cells" in current_measurement_dictionary:
            found_earfcns = [current_measurement_dictionary["current_earfcn"]]
            for neighbor in current_measurement_dictionary["neighbor_cells"]:
                if neighbor["n_earfcn"] not in found_earfcns:
                    self.neighbor_stack.append(neighbor)
                    found_earfcns.append(neighbor["n_earfcn"])
            self.number_of_measurements_in_batch = len(self.neighbor_stack)
            print("Neighbor stack: \n", self.neighbor_stack)
        else:
            self.neighbor_stack = []

        return measurement_result_batch_to_return


    @_machine.output()
    def activate_ok(self):
        print("activate to measure\n")
        self.state = "measure"

    @_machine.output()
    def measure_ok(self):
        print("measure to wait_measurement_result\n")
        self.state = "wait_measurement_result"

    @_machine.output()
    def wait_measurement_result_ok(self):
        print("wait_measurement_result to measure\n")
        self.state = "measure"



    @_machine.output()
    def add_to_measurement_batch(self, response):
        if response is not None:
            self.measurement_result_batch.append(response)


    @_machine.output()
    def clear_bandlock_ok(self):
        print("clear_bandlock to activate")
        self.state = "activate"


    @_machine.output()
    def get_clear_bandlock_command(self):
        return "AT%XBANDLOCK=0"

    clear_bandlock.upon(ok, enter=activate, outputs=[clear_bandlock_ok])
    clear_bandlock.upon(not_ok, enter=clear_bandlock, outputs=[])
    clear_bandlock.upon(get_next_command, enter=clear_bandlock, outputs=[get_clear_bandlock_command])

    activate.upon(ok, enter=measure, outputs=[activate_ok])
    activate.upon(not_ok, enter=activate, outputs=[])
    activate.upon(get_next_command, enter=activate, outputs=[get_activate_command])


    # measure.upon(ok, enter=wait_measurement_result, outputs=[measure_ok])
    # measure.upon(not_ok, enter=measure, outputs=[])
    measure.upon(get_next_command, enter=wait_measurement_result, outputs=[get_measure_command, measure_ok])

    wait_measurement_result.upon(ok, enter=measure, outputs=[save_measurement, wait_measurement_result_ok])
    wait_measurement_result.upon(not_ok, enter=measure, outputs=[])
    wait_measurement_result.upon(get_next_command, enter=wait_measurement_result, outputs=[get_empty_command])



