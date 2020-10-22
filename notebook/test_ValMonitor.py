from unittest import TestCase
from snippets  import ValMonitor

class TestValMonitor(TestCase):
    def test_time_to_save(self):
        vm = ValMonitor(2)
        vals =[5.0,4.0,3.0,2.0,3.0,5.0, 6.0]

        for i in range(4):
            should_save_model =vm.time_to_save(vals[i])
            self.assertEqual(True,should_save_model)

            should_stop_training = vm.time_to_stop()
            self.assertEqual(False, should_stop_training)
        for i in range(4,7):
            should_save_model = vm.time_to_save(vals[i])
            self.assertEqual(False, should_save_model)

            if i == 4:
                self.assertEqual(False, vm.time_to_stop())
            if i==5:
                self.assertEqual(True, vm.time_to_stop())
            if i==6:
                self.assertEqual(True, vm.time_to_stop())

