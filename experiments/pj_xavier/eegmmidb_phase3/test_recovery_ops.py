"""No-network safety and byte-identity tests for phase3 artifact recovery."""
from pathlib import Path
import hashlib,importlib.util,json,tempfile,unittest,zipfile
from unittest import mock
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('phase3_recovery_ops',Path(__file__).with_name('recovery_ops.py'))
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)

def fixture(root,content=b'exact scientific evidence',name='runs/run/checkpoint.pt'):
 root=Path(root);archive=root/'source.zip'
 with zipfile.ZipFile(archive,'w') as z:z.writestr(name,content)
 ledger={'files':{name:{'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}}}
 return archive,ledger

class RecoveryTests(unittest.TestCase):
 def test_roundtrip_and_exact_existing_member_preserved(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);dest=root/'dest';stage=root/'stage';dest.mkdir();stage.mkdir();archive,ledger=fixture(root)
   first=ops.extract_archive(archive,ledger,dest,stage);path=Path(first['runs/run/checkpoint.pt']['local_path']);before=path.stat().st_mtime_ns
   second=ops.extract_archive(archive,ledger,dest,stage)
   self.assertEqual(first,second);self.assertEqual(path.stat().st_mtime_ns,before)
 def test_conflicting_original_never_overwritten_and_remote_retained(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);dest=root/'dest';stage=root/'stage';dest.mkdir();stage.mkdir();archive,ledger=fixture(root)
   path=dest/'runs/run/checkpoint.pt';path.parent.mkdir(parents=True);path.write_bytes(b'user original')
   with self.assertRaisesRegex(ValueError,'retained separately'):ops.extract_archive(archive,ledger,dest,stage)
   self.assertEqual(path.read_bytes(),b'user original')
   self.assertEqual((stage/'conflicts/runs/run/checkpoint.pt').read_bytes(),b'exact scientific evidence')
   self.assertFalse(json.loads((stage/'conflict_receipt.json').read_text())['original_overwritten'])
 def test_path_escape_and_parent_symlink_refused(self):
  for name in ('../outside','/absolute','runs\\outside'):
   with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);dest=root/'dest';stage=root/'stage';dest.mkdir();stage.mkdir();archive,ledger=fixture(root,name=name)
    with self.assertRaises(ValueError):ops.extract_archive(archive,ledger,dest,stage)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);dest=root/'dest';stage=root/'stage';outside=root/'outside';dest.mkdir();stage.mkdir();outside.mkdir();(dest/'runs').symlink_to(outside,target_is_directory=True)
   archive,ledger=fixture(root)
   with self.assertRaises(ValueError):ops.extract_archive(archive,ledger,dest,stage)
   self.assertEqual(list(outside.iterdir()),[])
 def test_incorrect_byte_hash_refused(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);dest=root/'dest';stage=root/'stage';dest.mkdir();stage.mkdir();archive,ledger=fixture(root)
   ledger['files']['runs/run/checkpoint.pt']['sha256']='0'*64
   with self.assertRaisesRegex(ValueError,'member mismatch'):ops.extract_archive(archive,ledger,dest,stage)
   self.assertFalse((dest/'runs/run/checkpoint.pt').exists())
 def test_incomplete_existing_part_is_preserved(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);dest=root/'dest';stage=root/'stage';dest.mkdir();stage.mkdir();archive,ledger=fixture(root)
   path=dest/'runs/run/checkpoint.pt.recovery.part';path.parent.mkdir(parents=True);path.write_bytes(b'interrupted')
   with self.assertRaisesRegex(ValueError,'Preserve interrupted'):ops.extract_archive(archive,ledger,dest,stage)
   self.assertEqual(path.read_bytes(),b'interrupted')
 def test_postdeadline_transfer_blocked_but_stop_grace_bounded(self):
  with mock.patch.object(ops.time,'time',return_value=ops.DEADLINE+30):
   with self.assertRaisesRegex(ValueError,'original deadline'):ops.bounded_timeout(10)
   self.assertEqual(ops.stop_timeout(45,ops.DEADLINE+210),45)
   with self.assertRaises(ValueError):ops.stop_timeout(45,ops.DEADLINE+211)
 def test_postdeadline_override_cannot_run_arbitrary_remote_code(self):
  with mock.patch.object(ops.subprocess,'run') as run:
   with self.assertRaisesRegex(ValueError,'only the fixed terminal'):
    ops.remote('start_training()',stop_only_deadline=ops.DEADLINE+90)
   run.assert_not_called()
 def test_late_stop_requires_recovery_and_records_original_deadline(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);op=root/'ops';op.mkdir()
   terminal={'status':'terminal','owned_active_processes':[],'GPU_compute_processes':''}
   with mock.patch.object(ops,'PHASE',root),mock.patch.object(ops,'OPS',op),mock.patch.object(ops.time,'time',return_value=ops.DEADLINE+30),mock.patch.object(ops,'verify_recovery',return_value={'status':'passed','recovery_receipt_sha256':'a'*64}),mock.patch.object(ops,'remote',return_value=terminal) as query,mock.patch.object(ops.subprocess,'run',side_effect=[SimpleNamespace(returncode=0,stdout='stopped',stderr=''),SimpleNamespace(returncode=0,stdout='No active sessions',stderr='')]) as cli:
    result=ops.stop()
   self.assertEqual(result['status'],'stopped');self.assertEqual(result['original_compute_deadline_epoch'],ops.DEADLINE)
   self.assertEqual(result['stop_started_after_deadline_seconds'],30);self.assertFalse(result['postdeadline_training_or_download_authorized'])
   self.assertEqual(cli.call_args_list[0].args[0],[ops.CLI,'stop','--session',ops.SESSION])
   self.assertEqual(query.call_args.args[0],ops.TERMINAL_PROBE_SCRIPT)
 def test_remote_program_compiles_without_execution(self):
  source=ops.REMOTE_PACKAGE.replace('ATTESTATION',repr('/content/xavier_phase3/audit.json'))
  source='def package():\n'+'\n'.join(' '+line for line in source.splitlines())
  compile(source,'recovery-remote-program','exec')
  compile(ops.TERMINAL_PROBE_SCRIPT,'terminal-proof','exec')
  self.assertIn('split(bytes([0]))',ops.TERMINAL_PROBE_SCRIPT)

if __name__=='__main__':unittest.main(verbosity=2)
