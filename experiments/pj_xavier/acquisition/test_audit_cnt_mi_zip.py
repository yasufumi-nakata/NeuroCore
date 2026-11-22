import tempfile
from pathlib import Path
import struct
import unittest
import audit_cnt_mi_zip as a


def fixture(tag=0, bad_count=-525):
    n=16;width=4 if tag==1 else 2;samples=20;data=2100;ev=data+n*width*samples;length=0;nextfile=ev+9
    b=bytearray(nextfile+64);b[:12]=b'Version 3.0\0';struct.pack_into('<i',b,12,nextfile)
    struct.pack_into('<H',b,370,n);struct.pack_into('<H',b,376,250);struct.pack_into('<i',b,864,bad_count)
    struct.pack_into('<i',b,886,ev);struct.pack_into('<i',b,894,1)
    for i,name in enumerate(a.EOG+a.EEG):
        off=900+i*75;b[off:off+len(name)]=name.encode();struct.pack_into('<f',b,off+59,12.8);struct.pack_into('<f',b,off+71,1)
    struct.pack_into('<Bii',b,ev,2,length,0);b[nextfile+52]=tag
    return bytes(b)


class TestOriginalCNTLayout(unittest.TestCase):
    def test_present_original_tag_resolves_both_precisions_without_repair(self):
        for tag,width in [(0,2),(1,4)]:
            with tempfile.TemporaryDirectory() as t:
                p=Path(t)/'x.cnt';p.write_bytes(fixture(tag));h,ev=a.layout(p)
                self.assertEqual(h['scalar_bytes'],width);self.assertEqual(h['actual_sample_count_from_byte_span'],20)
                self.assertEqual(h['original_header_sample_count'],-525);self.assertFalse(h['source_sample_count_field_matches'])
                self.assertEqual(h['original_is32bit_tag'],tag);self.assertEqual(ev,b'')
                self.assertEqual(p.read_bytes(),fixture(tag))

    def test_absent_unknown_or_unlinked_tag_never_defaults_to_int16(self):
        good={'nextfile':3009,'event_offset':3000,'event_length':0,'file_size':3073,'tag':0}
        for key,value in [('tag',2),('nextfile',0),('nextfile',3010),('file_size',3050)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):a.precision_from_tag(**{**good,key:value})

    def test_positive_header_directly_contradicting_tag_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.cnt';p.write_bytes(fixture(0,bad_count=10))
            with self.assertRaisesRegex(ValueError,'contradicts'):a.layout(p)

    def test_raw_negative_event_offsets_are_not_clipped(self):
        self.assertEqual(a.untranslated_event_sample(2100,2,n_channels=16,scalar_bytes=2,data_offset=2100),(-1,0))
        self.assertEqual(a.untranslated_event_sample(2133,2,n_channels=16,scalar_bytes=2,data_offset=2100),(0,1))
        self.assertEqual(a.untranslated_event_sample(3100,2,n_channels=16,scalar_bytes=2,data_offset=2100),(30,8))


if __name__=='__main__':unittest.main()
