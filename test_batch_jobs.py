import base64
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from batch_jobs import BatchStore, BatchRunner, process_item


class BatchStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'batches.sqlite3'
        self.store = BatchStore(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_job_survives_restart_and_report_is_readable(self):
        batch_id = self.store.create('客户甲', '春季活动', '国内信息流', [
            dict(external_id='creative-1', text='本款帆布包售价69元。', mode='local', complete=True),
            dict(external_id='creative-2', text='全网第一。', mode='local', complete=True),
        ])
        running = self.store.claim()
        self.assertIsNotNone(running)
        restarted = BatchStore(self.path)
        self.assertEqual(restarted.summary(batch_id)['counts']['queued'], 2)
        job = restarted.claim()
        report = process_item(json.loads(job['payload']), lambda: {})
        restarted.finish(job['id'], report=report)
        detail = restarted.detail(batch_id, job['id'])
        self.assertEqual(detail['status'], 'succeeded')
        self.assertIn('findings', detail['report'])
        self.assertEqual(restarted.summary(batch_id)['counts']['succeeded'], 1)

    def test_failed_job_can_be_retried_once_queued(self):
        batch_id = self.store.create('客户甲', '活动', '国内信息流', [dict(text='全网第一。', mode='local')])
        job = self.store.claim()
        self.store.finish(job['id'], error='模拟失败')
        self.assertTrue(self.store.retry(batch_id, job['id']))
        self.assertFalse(self.store.retry(batch_id, job['id']))
        self.assertEqual(self.store.summary(batch_id)['counts']['queued'], 1)

    def test_same_image_in_batch_reuses_ocr(self):
        raw = b'example-image-bytes'
        uri = 'data:image/png;base64,' + base64.b64encode(raw).decode()
        material = dict(filename='first.png', page_count=1, pages=[dict(page=1, source_filename='first.png',
                        source_page=1, text='包邮', quality_issues=[], visual_notes='', review_notes=[])])
        payload = dict(text='本品69元。', evidence='', complete=True, mode='local',
                       images=[dict(filename='first.png', image=uri)])
        with patch('batch_jobs.extract_bundle', return_value=material) as extract:
            first = process_item(payload, lambda: {}, self.store, 'one-batch')
            payload['images'][0]['filename'] = 'second.png'
            second = process_item(payload, lambda: {}, self.store, 'one-batch')
        self.assertEqual(extract.call_count, 1)
        self.assertFalse(first['extraction_cache_hit'])
        self.assertTrue(second['extraction_cache_hit'])
        self.assertEqual(second['material']['pages'][0]['source_filename'], 'second.png')

    def test_image_batch_keeps_report_when_ai_quote_cannot_be_located(self):
        from reviewer import RULES
        raw = b'example-image-bytes'
        uri = 'data:image/png;base64,' + base64.b64encode(raw).decode()
        material = dict(filename='case01.png', page_count=1, pages=[dict(page=1,
            source_filename='case01.png', source_page=1, text='活动价 ¥69',
            quality_issues=[], visual_notes='价格位于图片左下角', review_notes=[])])
        checks = [dict(rule_id=rule['id'], status='未发现风险', quote='',
                       reason='模拟返回', suggestion='') for rule in RULES]
        checks[0].update(status='疑似风险', quote='活动价￥69', reason='价格需核对',
            issues=[dict(status='疑似风险', quote='活动价￥69', reason='价格需核对',
                         suggestion='核对原图', missing_evidence=[])])
        model = dict(scope=dict(status='in_scope', reason='商品宣传'), checks=checks)
        payload = dict(text='新款包包五折起', evidence='', complete=True, mode='ai',
                       images=[dict(filename='case01.png', image=uri)])
        with patch('batch_jobs.extract_bundle', return_value=material), \
             patch('materials.review_image_parts', return_value=[{'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}]), \
             patch('reviewer.call_json', return_value=model):
            report = process_item(payload, lambda: {'api_key':'test', 'base_url':'https://example.com'}, self.store, 'batch')
        self.assertEqual(report['review_model'], 'qwen-vl-plus')
        self.assertIn('引用待核对', report['conclusion'])
        self.assertEqual(report['findings'][0]['status'], '证据不足')
        self.assertEqual(report['material']['pages'][0]['text'], '活动价 ¥69')

    def test_background_runner_finishes_a_batch_without_browser_requests(self):
        batch_id = self.store.create('客户甲', '活动', '信息流', [
            dict(external_id=str(i), text=f'帆布包售价{i}元。', mode='local') for i in range(100)
        ])
        runner = BatchRunner(self.store, lambda: {}, threading.BoundedSemaphore(2), workers=2)
        runner.notify()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if self.store.summary(batch_id)['counts']['succeeded'] == 100:
                    break
                time.sleep(0.05)
            self.assertEqual(self.store.summary(batch_id)['counts']['succeeded'], 100)
        finally:
            runner.stop()

    def test_pending_limit_prevents_unbounded_model_queue(self):
        with patch('batch_jobs.MAX_PENDING_ITEMS', 1):
            self.store.create('客户甲', '活动', '信息流', [dict(text='素材甲', mode='local')])
            with self.assertRaisesRegex(ValueError, '最多同时排队'):
                self.store.create('客户甲', '活动', '信息流', [dict(text='素材乙', mode='local')])

    def test_cross_batch_queue_decision_history_and_source_asset(self):
        raw = b'example-image-bytes'
        uri = 'data:image/png;base64,' + base64.b64encode(raw).decode()
        batch_id = self.store.create('客户甲', '活动', '信息流', [dict(text='全网第一。', mode='local',
            images=[dict(filename='poster.png', image=uri)])])
        job = self.store.claim()
        report = dict(conclusion='待复核', findings=[dict(level='高', status='疑似风险')], blocked=True)
        self.store.finish(job['id'], report=report)
        queue = self.store.list_queue(search='客户甲', machine='succeeded', decision='pending', risk='high')
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]['job_id'], job['id'])
        self.assertNotIn('image', str(queue[0]))
        self.assertEqual(self.store.asset(batch_id, job['id'], 1)['content'], raw)
        self.assertTrue(self.store.record_decision(batch_id, job['id'], 'changes_requested', '审核员甲', '绝对化表述需修改'))
        self.assertTrue(self.store.record_decision(batch_id, job['id'], 'evidence_requested', '审核员乙', '请补充对应证明材料'))
        detail = self.store.detail(batch_id, job['id'])
        self.assertEqual(detail['decision']['decision'], 'evidence_requested')
        self.assertEqual(len(detail['review_history']), 2)
        self.assertEqual(self.store.list_queue(decision='pending'), [])


class BatchRouteTests(unittest.TestCase):
    def setUp(self):
        from app import app
        self.temp = tempfile.TemporaryDirectory()
        self.store = BatchStore(Path(self.temp.name) / 'batches.sqlite3')
        self.client = app.test_client()
        self.auth = {'X-Access-Code': 'test-access-code'}
        self.env = patch.dict('os.environ', {'APP_ACCESS_CODE': 'test-access-code'})
        self.store_patch = patch('app.batch_store', self.store)
        self.runner_patch = patch('app.batch_runner')
        self.env.start(); self.store_patch.start(); self.runner_patch.start()

    def tearDown(self):
        self.runner_patch.stop(); self.store_patch.stop(); self.env.stop()
        self.temp.cleanup()

    def test_api_batch_requires_code_for_reads_and_writes(self):
        body = dict(client='客户甲', campaign='活动', channel='信息流', consent=True,
                    items=[dict(external_id='creative-1', text='本款帆布包售价69元。', mode='local')])
        self.assertEqual(self.client.post('/api/batches', json=body).status_code, 401)
        created = self.client.post('/api/batches', json=body, headers=self.auth)
        self.assertEqual(created.status_code, 202)
        batch_id = created.json['batch_id']
        self.assertEqual(self.client.get('/api/batches/'+batch_id).status_code, 401)
        result = self.client.get('/api/batches/'+batch_id, headers=self.auth)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json['total'], 1)

    def test_home_is_unified_queue_and_manual_item_uses_it(self):
        home = self.client.get('/')
        self.assertEqual(home.status_code, 200)
        self.assertIn('手动补录', home.get_data(as_text=True))
        self.assertNotIn('返回单条审核', home.get_data(as_text=True))
        data = dict(client='客户甲', campaign='活动', channel='信息流', consent='true',
                    text='本款帆布包售价69元。', evidence='内部资料', complete='true')
        self.assertEqual(self.client.post('/api/batches/manual', data=data).status_code, 401)
        created = self.client.post('/api/batches/manual', data=data, headers=self.auth)
        self.assertEqual(created.status_code, 202)
        batch = self.client.get('/api/batches/' + created.json['batch_id'], headers=self.auth).json
        self.assertEqual(batch['total'], 1)
        self.assertEqual(batch['items'][0]['external_id'], '手动补录-1')
        payload = json.loads(self.store.claim()['payload'])
        self.assertEqual(payload['text'], data['text'])
        self.assertEqual(payload['evidence'], data['evidence'])

    def test_manual_upload_has_same_queue_limit_and_validates_file_type(self):
        base = dict(client='客户甲', campaign='活动', channel='信息流', consent='true')
        invalid = self.client.post('/api/batches/manual', data={**base, 'files': (io.BytesIO(b'wrong'), 'x.exe')}, headers=self.auth)
        self.assertEqual(invalid.status_code, 400)
        created = self.client.post('/api/batches/manual', data={**base, 'files': (io.BytesIO(b'example'), 'poster.png')}, headers=self.auth)
        self.assertEqual(created.status_code, 202)
        payload = json.loads(self.store.claim()['payload'])
        self.assertEqual(payload['images'][0]['filename'], 'poster.png')

    def test_reviewer_routes_require_code_and_completed_machine_review(self):
        data = dict(client='客户甲', campaign='活动', channel='信息流', consent='true', text='全网第一。')
        created = self.client.post('/api/batches/manual', data=data, headers=self.auth).json
        batch_id = created['batch_id']
        job_id = self.store.summary(batch_id)['items'][0]['id']
        self.assertEqual(self.client.get('/api/batches/queue').status_code, 401)
        self.assertEqual(self.client.get('/api/batches/queue', headers=self.auth).status_code, 200)
        path = f'/api/batches/{batch_id}/items/{job_id}/decision'
        body = dict(decision='approved', reviewer='审核员甲', reason='人工核对已完成')
        self.assertEqual(self.client.post(path, json=body, headers=self.auth).status_code, 400)
        job = self.store.claim()
        self.store.finish(job['id'], report=dict(conclusion='待人工确认', findings=[], blocked=False))
        self.assertEqual(self.client.post(path, json=body).status_code, 401)
        approved = self.client.post(path, json=body, headers=self.auth)
        self.assertEqual(approved.status_code, 200)
        self.assertEqual(approved.json['item']['decision']['decision'], 'approved')

    def test_source_asset_requires_code_and_correct_batch(self):
        base = dict(client='客户甲', campaign='活动', channel='信息流', consent='true')
        created = self.client.post('/api/batches/manual', data={**base, 'files': (io.BytesIO(b'image-bytes'), 'poster.png')}, headers=self.auth).json
        batch_id = created['batch_id']
        job_id = self.store.summary(batch_id)['items'][0]['id']
        path = f'/api/batches/{batch_id}/items/{job_id}/assets/1'
        self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.get(path, headers=self.auth).data, b'image-bytes')
        self.assertEqual(self.client.get(f'/api/batches/wrong/items/{job_id}/assets/1', headers=self.auth).status_code, 404)

    def test_batch_rejects_duplicate_ids_and_inline_images(self):
        base = dict(client='客户甲', campaign='活动', channel='信息流', consent=True)
        duplicate = self.client.post('/api/batches', json={**base, 'items': [dict(external_id='same', text='甲'), dict(external_id='same', text='乙')]}, headers=self.auth)
        self.assertEqual(duplicate.status_code, 400)
        inline = self.client.post('/api/batches', json={**base, 'items': [dict(text='甲', images=[{'image':'data:image/png;base64,a'}])]}, headers=self.auth)
        self.assertEqual(inline.status_code, 400)

    def test_excel_route_accepts_more_than_old_browser_limit(self):
        from test_excel_import import fixture
        rows = {row: {'A': f'素材{row}'} for row in range(2, 13)}
        data = dict(client='客户甲', campaign='活动', channel='信息流', consent='true',
                    file=(io.BytesIO(fixture(rows)), 'batch.xlsx'))
        response = self.client.post('/api/batches/excel', data=data, headers=self.auth)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json['total'], 11)


if __name__ == '__main__':
    unittest.main()
