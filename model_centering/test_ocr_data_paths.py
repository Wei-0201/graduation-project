import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import sign_ocr_engine as engine


class EnglishDataTests(unittest.TestCase):
    def test_existing_language_file_path_is_discovered_without_trusting_wrong_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td)/'eng.traineddata').write_bytes(b'x'*2048)
            with patch.dict(os.environ, {'CAR_TESSDATA_DIR': td, 'TESSDATA_PREFIX': 'nonexistent/tessdata/5'}):
                paths = engine.data_directories('mock_tesseract')
                self.assertIn(os.path.realpath(td), paths)
                self.assertNotIn(os.path.realpath('nonexistent/tessdata/5'), paths)

    def test_empty_language_file_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td)/'eng.traineddata').write_bytes(b'')
            with patch.dict(os.environ, {'CAR_TESSDATA_DIR': td}):
                self.assertNotIn(os.path.realpath(td), engine.data_directories('mock_tesseract'))

    def test_loader_really_invokes_ocr_and_passes_explicit_tessdata_directory(self):
        ok = subprocess.CompletedProcess([], 0, stdout=b'TEST', stderr=b'')
        with patch.object(engine, '_resolved', None), \
                patch.object(engine, 'executable', return_value='mock_tesseract'), \
                patch.object(engine, 'data_directories', return_value=['/actual/eng/data']), \
                patch.object(engine.subprocess, 'run', return_value=ok) as run, \
                patch.dict(os.environ, {'TESSDATA_PREFIX': '/wrong/5/data'}):
            self.assertIn('/actual/eng/data', engine.check_available())
            args, kwargs = run.call_args
            self.assertIn('stdin', args[0])
            self.assertEqual(args[0][args[0].index('--tessdata-dir')+1], '/actual/eng/data')
            self.assertNotIn('--list-langs', args[0])
            self.assertTrue(kwargs['input'])
            self.assertNotIn('TESSDATA_PREFIX', kwargs['env'])

    def test_wrong_version_data_falls_back_to_loadable_data(self):
        fail = subprocess.CompletedProcess([], 1, stdout=b'', stderr=b'Failed loading language eng')
        ok = subprocess.CompletedProcess([], 0, stdout=b'TEST', stderr=b'')
        with patch.object(engine, '_resolved', None), \
                patch.object(engine, 'executable', return_value='mock_tesseract'), \
                patch.object(engine, 'data_directories', return_value=['/bad/5', '/good/4.00']), \
                patch.object(engine.subprocess, 'run', side_effect=[fail, ok]):
            self.assertIn('/good/4.00', engine.check_available())
            self.assertEqual(engine._resolved, ('mock_tesseract', '/good/4.00'))

    def test_language_failure_text_is_rejected_even_with_zero_exit(self):
        bad = subprocess.CompletedProcess([], 0, stdout=b'', stderr=b'Could not initialize tesseract.')
        self.assertTrue(engine.engine_error(bad))

    def test_failed_probe_cannot_report_ready(self):
        bad = subprocess.CompletedProcess([], 1, stdout=b'eng', stderr=b'Failed loading language eng')
        with patch.object(engine, '_resolved', None), \
                patch.object(engine, 'executable', return_value='mock_tesseract'), \
                patch.object(engine, 'data_directories', return_value=[]), \
                patch.object(engine.subprocess, 'run', return_value=bad):
            with self.assertRaises(RuntimeError):
                engine.check_available()
            self.assertIsNone(engine._resolved)

    def test_live_calls_reuse_verified_data_directory_and_remove_bad_inherited_prefix(self):
        ok = subprocess.CompletedProcess([], 0, stdout=b'STOP', stderr=b'')
        with patch.object(engine, '_resolved', ('mock_tesseract', '/verified/4.00')), \
                patch.object(engine.subprocess, 'run', return_value=ok) as run, \
                patch.dict(os.environ, {'TESSDATA_PREFIX': '/wrong/5'}):
            text = engine.image_to_string(np.zeros((20, 20), np.uint8), '--psm 7')
            self.assertEqual(text, 'STOP')
            args, kwargs = run.call_args
            self.assertIn('/verified/4.00', args[0])
            self.assertNotIn('TESSDATA_PREFIX', kwargs['env'])


if __name__ == '__main__':
    unittest.main()
