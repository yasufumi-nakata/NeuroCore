import unittest
from eeglab_action_probe import class_of_press,count_preaction_windows

class TestEventTypes(unittest.TestCase):
    def test_actions(self):
        self.assertEqual(class_of_press('key pressed'),'keyboard')
        self.assertEqual(class_of_press('SPACE pressed'),'keyboard')
        self.assertEqual(class_of_press('MouseButtonLeft pressed'),'mouse_click')
        self.assertEqual(class_of_press('MouseWheelUp120 pressed'),'mouse_scroll')
        self.assertIsNone(class_of_press('MouseButtonLeft released'))
    def test_prior_event_guard(self):
        ev=[(2.0,'MouseButtonLeft pressed'),(2.2,'MouseButtonLeft released'),(4.0,'key pressed'),(8.0,'key pressed')]
        out=count_preaction_windows(ev,10)
        self.assertEqual(out['targets_by_class'],{'mouse_click':1,'keyboard':2})
        self.assertEqual(out['history_with_no_other_recorded_events'],2)
    def test_no_future_goal_claim(self):
        out=count_preaction_windows([(3,'key pressed')],10)
        self.assertIn('not free-form user intent',out['interpretation'])

if __name__=='__main__':unittest.main()
