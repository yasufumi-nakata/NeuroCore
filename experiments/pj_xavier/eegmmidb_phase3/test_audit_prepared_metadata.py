import unittest
from fractions import Fraction
import audit_prepared_metadata as a
class TestIntervalAccounting(unittest.TestCase):
 def test_overlap_counted_once_and_touching_intervals_merge(self):
  self.assertEqual(a.union_length([(0,30),(20,40),(40,50),(100,110)]),60)
 def test_exact_fractional_grid_union(self):
  self.assertEqual(a.union_length([(Fraction(5),Fraction(8)),(Fraction('5.2'),Fraction('8.2'))]),Fraction('3.2'))
 def test_raw_native_indices_are_integer_not_bool_or_float(self):
  for x in [True,2.,'2']:
   with self.subTest(x=x),self.assertRaisesRegex(ValueError,'Integer'):a.require_integer(x)
  a.require_integer(2)
 def test_test_and_action_metadata_fail_before_array_use(self):
  with self.assertRaisesRegex(ValueError,'Test'):a.validate_row({'split':'test'},{},set(),set(),{}, {})
  with self.assertRaisesRegex(ValueError,'Annotation'):a.validate_row({'split':'train','label':'left'},{},set(),set(),{}, {})
if __name__=='__main__':unittest.main()
