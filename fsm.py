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

    @_machine.state()
    def deactivate(self):
        "In this state, modem deactivation command is sent."

    @_machine.state()
    def adjust_search_params(self):
        "In this state, modem search settings are adjusted."

    @_machine.state()
    def activate_after_adjustment(self):
        "In this state, modem search settings are adjusted."

    @_machine.state()
    def adjusted_measure(self):
        "In this state, modem sends measure command with adjusted settings."

    @_machine.state()
    def wait_adjusted_measurement_result(self):
        ""

    @_machine.state()
    def clear_search_params(self):
        ""

    @_machine.state()
    def cpsms_before_measure(self):
        ""

    @_machine.state()
    def reset_cpsms_before_measure(self):
        ""

    @_machine.state()
    def cpsms_before_adjusted_measure(self):
        ""

    @_machine.state()
    def reset_cpsms_before_adjusted_measure(self):
        ""


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

    @_machine.input()
    def ok_neighbor_stack_0(self):
        "return next command"

    @_machine.input()
    def ok_neighbor_stack_larger_0(self):
        "return next command"

    @_machine.output()
    def get_activate_command(self):
        return "AT+CFUN=1"

    @_machine.output()
    def get_deactivate_command(self):
        return "AT+CFUN=0"

    @_machine.output()
    def get_measure_command(self):
        return "AT%NCELLMEAS"

    @_machine.output()
    def get_empty_command(self):
        return ""

    @_machine.output()
    def get_adjust_search_params_command(self):
        if self.adjust_search_params_not_ok:
            self.adjust_search_params_not_ok = False
            return self.last_adjust_search_params_command

        neighbor = self.neighbor_stack.pop(0)
        command = 'AT%XBANDLOCK=2,"{}"'.format(int2onehot(get_band(int(neighbor["n_earfcn"]))))
        self.last_adjust_search_params_command = command
        return command

    @_machine.output()
    def adjust_search_params_not_ok(self):
        self.adjust_search_params_not_ok = True

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
    def change_state_based_on_neighbor_count(self):
        if len(self.neighbor_stack) == 0:
            self.ok_neighbor_stack_0()
        else:
            self.ok_neighbor_stack_larger_0()

    @_machine.output()
    def get_clear_search_params_command(self):
        return "AT%XBANDLOCK=0"

    @_machine.output()
    def activate_ok(self):
        print("activate to cpsms_before_measure\n")
        self.state = "cpsms_before_measure"

    @_machine.output()
    def measure_ok(self):
        print("measure to wait_measurement_result\n")
        self.state = "wait_measurement_result"

    @_machine.output()
    def wait_measurement_result_ok(self):
        print("wait_measurement_result to deactivate\n")
        self.state = "deactivate"

    @_machine.output()
    def deactivate_ok(self):
        print("deactivate to clear_search_params\n")
        self.state = "clear_search_params"

    @_machine.output()
    def clear_search_params_0_ok(self):
        print("clear_search_params to activate\n")
        self.state = "activate"

    @_machine.output()
    def clear_search_params_larger_0_ok(self):
        print("clear_search_params to adjust_search_params\n")
        self.state = "adjust_search_params"

    @_machine.output()
    def adjust_search_params_ok(self):
        print("adjust_search_params to activate_after_adjustment\n")
        self.state = "activate_after_adjustment"

    @_machine.output()
    def activate_after_adjustment_ok(self):
        print("activate_after_adjustment to reset_cpsms_before_adjusted_measure\n")
        self.state = "reset_cpsms_before_adjusted_measure"

    @_machine.output()
    def adjusted_measure_ok(self):
        print("adjusted_measure to wait_adjusted_measurement_result\n")
        self.state = "wait_adjusted_measurement_result"

    @_machine.output()
    def wait_adjusted_measurement_result_ok(self):
        print("wait_adjusted_measurement_result to deactivate\n")
        self.state = "deactivate"

    @_machine.output()
    def clear_search_params_ok(self):
        print("clear_search_params to deactivate\n")
        self.state = "deactivate"

    @_machine.output()
    def add_to_measurement_batch(self, response):
        if response is not None:
            self.measurement_result_batch.append(response)

    @_machine.output()
    def cpsms_before_measure_ok(self):
        print("cpsms_before_measure to reset_cpsms_before_measure")
        self.state = "reset_cpsms_before_measure"

    @_machine.output()
    def get_cpsms_command(self):
        return 'AT+CPSMS=1'

    @_machine.output()
    def cpsms_before_adjusted_measure_ok(self):
        print("cpsms_before_measure to reset_cpsms_before_adjusted_measure")
        self.state = "reset_cpsms_before_adjusted_measure"


    @_machine.output()
    def get_reset_cpsms_command(self):
        return "AT+CPSMS="

    @_machine.output()
    def reset_cpsms_before_measure_ok(self):
        print("reset_cpsms_before_measure to measure")
        self.state = "measure"

    @_machine.output()
    def reset_cpsms_before_adjusted_measure_ok(self):
        print("reset_cpsms_before_adjusted_measure to adjusted_measure")
        self.state = "adjusted_measure"

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

    activate.upon(ok, enter=cpsms_before_measure, outputs=[activate_ok])
    activate.upon(not_ok, enter=activate, outputs=[])
    activate.upon(get_next_command, enter=activate, outputs=[get_activate_command])

    cpsms_before_measure.upon(ok, enter=reset_cpsms_before_measure, outputs=[cpsms_before_measure_ok])
    cpsms_before_measure.upon(not_ok, enter=cpsms_before_measure, outputs=[])
    cpsms_before_measure.upon(get_next_command, enter=cpsms_before_measure, outputs=[get_cpsms_command])

    reset_cpsms_before_measure.upon(ok, enter=measure, outputs=[reset_cpsms_before_measure_ok])
    reset_cpsms_before_measure.upon(not_ok, enter=reset_cpsms_before_measure, outputs=[])
    reset_cpsms_before_measure.upon(get_next_command, enter=reset_cpsms_before_measure, outputs=[get_reset_cpsms_command])

    # measure.upon(ok, enter=wait_measurement_result, outputs=[measure_ok])
    # measure.upon(not_ok, enter=measure, outputs=[])
    measure.upon(get_next_command, enter=wait_measurement_result, outputs=[get_measure_command, measure_ok])

    wait_measurement_result.upon(ok, enter=deactivate, outputs=[save_measurement, wait_measurement_result_ok])
    wait_measurement_result.upon(not_ok, enter=measure, outputs=[])
    wait_measurement_result.upon(get_next_command, enter=wait_measurement_result, outputs=[get_empty_command])

    # deactivate.upon(ok, enter=deactivate, outputs=[change_state_based_on_neighbor_count])
    # deactivate.upon(ok_neighbor_stack_0, enter=activate, outputs=[deactivate_0_ok])
    # deactivate.upon(ok_neighbor_stack_larger_0, enter=adjust_search_params, outputs=[deactivate_larger_0_ok])
    # deactivate.upon(not_ok, enter=deactivate, outputs=[])
    # deactivate.upon(get_next_command, enter=deactivate, outputs=[get_deactivate_command])
    #
    # clear_search_params.upon(ok, enter=deactivate, outputs=[clear_search_params_ok])
    # clear_search_params.upon(not_ok, enter=clear_search_params, outputs=[])
    # clear_search_params.upon(get_next_command, enter=clear_search_params, outputs=[get_clear_search_params_command])

    clear_search_params.upon(ok, enter=clear_search_params, outputs=[change_state_based_on_neighbor_count])
    clear_search_params.upon(ok_neighbor_stack_0, enter=activate, outputs=[clear_search_params_0_ok])
    clear_search_params.upon(ok_neighbor_stack_larger_0, enter=adjust_search_params, outputs=[clear_search_params_larger_0_ok])
    clear_search_params.upon(not_ok, enter=clear_search_params, outputs=[])
    clear_search_params.upon(get_next_command, enter=clear_search_params, outputs=[get_clear_search_params_command])

    deactivate.upon(ok, enter=clear_search_params, outputs=[deactivate_ok])
    deactivate.upon(not_ok, enter=deactivate, outputs=[])
    deactivate.upon(get_next_command, enter=deactivate, outputs=[get_deactivate_command])

    adjust_search_params.upon(ok, enter=activate_after_adjustment, outputs=[adjust_search_params_ok])
    adjust_search_params.upon(not_ok, enter=adjust_search_params, outputs=[adjust_search_params_not_ok])
    adjust_search_params.upon(get_next_command, enter=adjust_search_params, outputs=[get_adjust_search_params_command])

    activate_after_adjustment.upon(ok, enter=cpsms_before_adjusted_measure, outputs=[activate_after_adjustment_ok])
    activate_after_adjustment.upon(not_ok, enter=activate_after_adjustment, outputs=[])
    activate_after_adjustment.upon(get_next_command, enter=activate_after_adjustment, outputs=[get_activate_command])

    cpsms_before_adjusted_measure.upon(ok, enter=reset_cpsms_before_adjusted_measure, outputs=[cpsms_before_adjusted_measure_ok])
    cpsms_before_adjusted_measure.upon(not_ok, enter=cpsms_before_adjusted_measure, outputs=[])
    cpsms_before_adjusted_measure.upon(get_next_command, enter=cpsms_before_adjusted_measure, outputs=[get_cpsms_command])

    reset_cpsms_before_adjusted_measure.upon(ok, enter=adjusted_measure, outputs=[reset_cpsms_before_adjusted_measure_ok])
    reset_cpsms_before_adjusted_measure.upon(not_ok, enter=reset_cpsms_before_adjusted_measure, outputs=[])
    reset_cpsms_before_adjusted_measure.upon(get_next_command, enter=reset_cpsms_before_adjusted_measure,
                                       outputs=[get_reset_cpsms_command])

    # adjusted_measure.upon(ok, enter=wait_adjusted_measurement_result, outputs=[adjusted_measure_ok])
    # adjusted_measure.upon(not_ok, enter=adjusted_measure, outputs=[])
    adjusted_measure.upon(get_next_command, enter=wait_adjusted_measurement_result, outputs=[get_measure_command, adjusted_measure_ok])

    wait_adjusted_measurement_result.upon(ok, enter=deactivate, outputs=[wait_adjusted_measurement_result_ok, add_to_measurement_batch])
    wait_adjusted_measurement_result.upon(not_ok, enter=adjusted_measure, outputs=[])
    wait_adjusted_measurement_result.upon(get_next_command, enter=wait_adjusted_measurement_result, outputs=[get_empty_command])



# controller = Controller()
#
# print(controller.get_next_command())
# controller.ok()
#
# print(controller.get_next_command())
# controller.ok()
#
#
# print(controller.get_next_command())
# controller.ok('%NCELLMEAS: 0,"02A5C002","26201","57F7",65535,1300,364,33,14,2475947,6400,167,41,20,31,6400,72,35,8,31,6400,390,34,6,31,0')
#
# print(controller.get_next_command())
# controller.ok()
#
# print(controller.get_next_command())
# controller.ok('%NCELLMEAS: 0,"02A5C002","26201","57F7",65535,1300,364,33,14,2475947,6400,167,41,20,31,6400,72,35,8,31,6400,390,34,6,31,0')
#
#
# print(controller.get_next_command())
# controller.ok()
#
# print(controller.get_next_command())
# controller.ok()
#
# print(controller.get_next_command())
# controller.ok('%NCELLMEAS: 0,"02A5C002","26201","57F7",65535,1300,364,34,23,2471467,0')
#
# print(controller.get_next_command())
# controller.ok('%NCELLMEAS: 0,"02A5C002","26201","57F7",65535,1300,364,34,23,2471467,0')
#
#
# print(controller.get_next_command())
# controller.ok()
#
#
# print(controller.get_next_command())
# controller.not_ok()
# controller.ok()
#
# print(controller.get_next_command())
# controller.ok()
#
#
# print(controller.get_next_command())
# controller.ok()
#
# print(controller.get_next_command())
# controller.ok()
#
# print(controller.get_next_command())
# controller.ok()
#
# print(controller.get_next_command())
# controller.ok()
#
# print(controller.get_next_command())
# controller.ok('%NCELLMEAS: 0,"02A5C002","26201","57F7",65535,1300,364,34,23,2471467,0')