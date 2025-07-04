# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Type

import gradio as gr

from swift.llm import ModelType
from swift.llm.model.register import get_all_models
from swift.ui.base import BaseUI


class RefModel(BaseUI):
    group = 'llm_grpo'

    locale_dict = {
        'ref_tab': {
            'label': {
                'zh': 'Ref模型参数设置',
                'en': 'Ref model parameters settings'
            },
        },
        'ref_model_type': {
            'label': {
                'zh': 'Ref模型类型',
                'en': 'Ref model type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Model type supported by SWIFT'
            }
        },
        'ref_model': {
            'label': {
                'zh': 'Ref模型id或路径',
                'en': 'Ref model id or path'
            },
            'info': {
                'zh': '实际的模型id或路径',
                'en': 'The actual model id or path'
            }
        },
        'sync_ref_model': {
            'label': {
                'zh': '同步ref模型',
                'en': 'Ref model synchronization'
            },
            'info': {
                'zh': '是否定期同步ref模型',
                'en': 'Whether to synchronize ref model'
            }
        },
        'ref_model_sync_steps': {
            'label': {
                'zh': '同步频率',
                'en': 'Sync steps'
            },
            'info': {
                'zh': 'Ref模型同步频率',
                'en': 'Ref model synchronization frequency'
            }
        },
        'ref_model_mixup_alpha': {
            'label': {
                'zh': '混合系数',
                'en': 'Mixup alpha'
            },
            'info': {
                'zh': '控制在更新过程中model和先前ref model之间的混合',
                'en': 'Controls the blending between model and the previous ref_model during the update process'
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='ref_tab', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(
                        elem_id='ref_model', scale=20, value=None, choices=get_all_models(), allow_custom_value=True)
                    gr.Dropdown(elem_id='ref_model_type', choices=ModelType.get_model_name_list(), value=None, scale=20)
                with gr.Row():
                    gr.Checkbox(elem_id='sync_ref_model', scale=4)
                    gr.Textbox(elem_id='ref_model_sync_steps', lines=1, value=512, scale=4)
                    gr.Slider(elem_id='ref_model_mixup_alpha', minimum=0.0, maximum=1.0, step=0.05, value=0.6, scale=4)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('ref_model').change(
            partial(cls.update_input_model, allow_keys=['ref_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('ref_model')],
            outputs=[cls.element('ref_model_type')])
