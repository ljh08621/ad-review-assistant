import io
import os
import unittest
from unittest.mock import patch
from PIL import Image
import pymupdf
from app import app, signer, recent_calls
from materials import images_from_file, extract, extract_bundle, review_image_parts
from reviewer import review, check_ai_payload, RULES


class MaterialTests(unittest.TestCase):
    def setUp(self):
        recent_calls.clear()
        self.client=app.test_client()

    def image(self):
        output=io.BytesIO();Image.new('RGB',(800,600),'white').save(output,format='PNG');return output.getvalue()

    def pdf(self,pages):
        with pymupdf.open() as doc:
            for _ in range(pages):doc.new_page()
            return doc.tobytes()

    def test_renders_all_pdf_pages(self):
        self.assertEqual(len(images_from_file('test.pdf',self.pdf(2))),2)

    def test_direct_review_receives_every_pdf_page_as_image(self):
        parts = review_image_parts([('poster.pdf', self.pdf(2))])
        self.assertEqual([part['type'] for part in parts], ['text','image_url','text','image_url'])
        self.assertIn('第 2 页', parts[2]['text'])
        self.assertTrue(parts[3]['image_url']['url'].startswith('data:image/'))

    def test_pdf_over_limit_rejected(self):
        with self.assertRaisesRegex(ValueError,'1～4'):images_from_file('test.pdf',self.pdf(5))

    def test_fake_image_rejected(self):
        with self.assertRaises(ValueError):images_from_file('test.png',b'not an image')

    def test_image_extract_mocked(self):
        with patch('materials.call_json',return_value={'text':'治愈','quality_issues':['小字模糊'],'visual_notes':'标题醒目'}):
            result=extract('test.png',self.image(),{})
        self.assertEqual(result['page_count'],1)
        self.assertEqual(result['pages'][0]['quality_issues'],['小字模糊'])

    def test_unreadable_material_forces_a10(self):
        r=review({'text':'普通文案','complete':True},{'_quality_issues':['小字模糊']})
        self.assertIn('A-10',{f['rule_id'] for f in r['findings']})

    def test_access_code(self):
        with patch.dict(os.environ,{'APP_ACCESS_CODE':'test-access-1234'}):
            self.assertEqual(self.client.post('/api/review',json={'text':'测试'}).status_code,401)
            self.assertEqual(self.client.post('/api/review',json={'text':'测试'},headers={'X-Access-Code':'test-access-1234'}).status_code,200)

    def test_secret_not_served(self):
        self.assertEqual(self.client.get('/config.json').status_code,404)

    def test_tampered_material_rejected(self):
        self.assertEqual(self.client.post('/api/review',json={'text':'测试','material_token':'fake'}).status_code,400)

    def test_signed_material_preserves_quality(self):
        material={'filename':'test.pdf','page_count':1,'pages':[{'page':1,'text':'无副作用','quality_issues':['模糊'],'visual_notes':'小字'}]}
        r=self.client.post('/api/review',json={'text':'','material_token':signer.dumps(material),'complete':True,'extraction_confirmed':True}).get_json()
        self.assertTrue(r['blocked'])
        self.assertIn('A-10',{f['rule_id'] for f in r['findings']})
        self.assertIn('第1页',next(f['location'] for f in r['findings'] if f['rule_id']=='A-06'))

    def test_cross_origin_rejected(self):
        self.assertEqual(self.client.post('/api/review',json={'text':'测试'},headers={'Origin':'https://example.invalid'}).status_code,403)

    def test_ai_consent_required(self):
        self.assertEqual(self.client.post('/api/review',json={'text':'测试','mode':'ai'}).status_code,400)

    def test_upload_consent_required(self):
        response=self.client.post('/api/extract',data={'file':(io.BytesIO(self.image()),'test.png')})
        self.assertEqual(response.status_code,400)

    def test_correction_keeps_original(self):
        material={'filename':'test.pdf','page_count':1,'pages':[{'page':1,'text':'满意度98%','quality_issues':[],'visual_notes':'文字'}]}
        r=self.client.post('/api/review',json={'text':'','material_token':signer.dumps(material),'complete':True,'extraction_confirmed':True,'page_corrections':['满意率98%']}).get_json()
        self.assertEqual(r['material']['pages'][0]['original_ocr_text'],'满意度98%')
        self.assertEqual(r['material']['pages'][0]['text'],'满意率98%')
        self.assertNotIn('A-10',{f['rule_id'] for f in r['findings']})
        self.assertTrue(r['notices'])

    def test_clear_unconfirmed_image_is_notice(self):
        material={'filename':'test.png','page_count':1,'pages':[{'page':1,'text':'普通文案','quality_issues':[],'visual_notes':'清晰'}]}
        r=self.client.post('/api/review',json={'text':'','material_token':signer.dumps(material),'complete':True,'extraction_confirmed':False}).get_json()
        self.assertNotIn('A-10',{f['rule_id'] for f in r['findings']})
        self.assertIn('尚未人工核对',r['notices'][0])

    def test_bundle_single_call_and_page_sources(self):
        pages=[dict(page=i,text=f'内容{i}',quality_issues=[],visual_notes='清晰') for i in (1,2)]
        with patch('materials.call_json',return_value={'pages':pages}) as call:
            result=extract_bundle([('a.png',self.image()),('b.png',self.image())],{})
        self.assertEqual(call.call_count,1)
        self.assertEqual(result['pages'][1]['source_filename'],'b.png')
        self.assertEqual(result['page_count'],2)

    def test_bundle_missing_page_rejected(self):
        with patch('materials.call_json',return_value={'pages':[]}):
            with self.assertRaisesRegex(ValueError,'全部页面'):
                extract_bundle([('a.png',self.image())],{})

    def test_bundle_page_limit_before_model(self):
        with patch('materials.call_json') as call:
            with self.assertRaises(ValueError):
                extract_bundle([('a.pdf',self.pdf(3)),('b.pdf',self.pdf(2))],{})
            call.assert_not_called()

    def test_text_and_images_combined(self):
        material={'filename':'a.png / b.png','page_count':2,'pages':[dict(page=i,source_filename=f'{i}.png',source_page=1,text=f'图片{i}',quality_issues=[],visual_notes='清晰') for i in (1,2)]}
        with patch('app.review',return_value={'findings':[], 'completeness':'完整'}) as call:
            response=self.client.post('/api/review',json={'text':'配套文案','material_token':signer.dumps(material),'complete':True})
        self.assertEqual(response.status_code,200)
        text=call.call_args.args[0]['text']
        for expected in ('配套文案','图片1','图片2','1.png','2.png'):
            self.assertIn(expected,text)

    def test_resize_is_only_notice(self):
        output=io.BytesIO();Image.new('RGB',(2300,800),'white').save(output,format='PNG')
        with patch('materials.call_json',return_value={'text':'普通商品','quality_issues':[],'visual_notes':'文字清晰'}):
            r=extract('large.png',output.getvalue(),{})
        self.assertEqual(r['pages'][0]['quality_issues'],[])
        self.assertTrue(r['pages'][0]['review_notes'])

    def test_multiple_issues_one_rule(self):
        checks=[dict(rule_id=r['id'],status='不适用',quote='',reason='未触发',suggestion='',issues=[]) for r in RULES]
        checks[0].update(status='疑似风险',issues=[dict(status='疑似风险',quote=q,reason='缺依据',suggestion='补证据',missing_evidence=['依据']) for q in ['第一','唯一']])
        _,items=check_ai_payload({'checks':checks},'第一，唯一')
        self.assertEqual(len(items),2)


if __name__=='__main__':unittest.main()
