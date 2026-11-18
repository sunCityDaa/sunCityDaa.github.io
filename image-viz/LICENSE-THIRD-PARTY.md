# 第三方许可证与声明 / Third-party licenses and notices

本目录（`static/image-viz/`）中的代码由本博客编写。页面展示的数据来自两个第三方模型在 A100 上的一次真实推理。

**重要：本仓库不包含、也不分发任何模型权重。** `data/` 下只有生成出来的图像、张量形状与参数量等结构信息，以及激活/注意力的统计量。模型的输出（图像）按各自的许可证不属于模型的"衍生物"。

## 1. Qwen-Image-2.1

来源：<https://huggingface.co/Qwen/Qwen-Image-2.1>

许可证：**Qwen RESEARCH LICENSE AGREEMENT**（Release Date: September 20, 2026），仅限非商业的研究/评估用途。

按协议第 3(c) 条要求的声明 / Attribution notice required by Section 3(c)：

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

协议第 4(b) 条：若使用其输出创建、训练、微调或改进另一个对外提供的人工智能模型，须在产品文档中显著标注 "Built with Qwen" 或 "Improved using Qwen"。**本站未做任何此类用途**，只是把一次推理的过程可视化。

协议第 4(c) 条：不得以 "Qwen" 作为衍生物或产品的主要名称。本站此处仅作描述性使用（"由 Qwen-Image-2.1 生成"）。

完整协议文本（随模型一起分发于 `Qwen-Image-2.1/LICENSE`）：

```
Qwen RESEARCH LICENSE AGREEMENT
Release Date: September 20, 2026

By clicking to agree or by using or distributing any portion or element of the Qwen
Materials, you will be deemed to have recognized and accepted the content of this
Agreement, which is effective immediately.

1. Definitions
    a. This Qwen RESEARCH LICENSE AGREEMENT (this "Agreement") shall mean the terms and
       conditions for use, reproduction, distribution and modification of the Materials
       as defined by this Agreement.
    b. "We" (or "Us") shall mean Hangzhou Tongyi Laboratory Technology Co., Ltd.
    c. "You" (or "Your") shall mean a natural person or legal entity exercising the
       rights granted by this Agreement and/or using the Materials for any purpose and
       in any field of use.
    d. "Third Parties" shall mean individuals or legal entities that are not under
       common control with us or you.
    e. "Qwen" shall mean the large language models, diffusion models, and software and
       algorithms, consisting of trained model weights, parameters (including optimizer
       states), machine-learning model code, inference-enabling code, training-enabling
       code, fine-tuning enabling code and other elements of the foregoing distributed
       by us.
    f. "Materials" shall mean, collectively, our proprietary Qwen and Documentation (and
       any portion thereof) made available under this Agreement.
    g. "Source" form shall mean the preferred form for making modifications, including
       but not limited to model source code, documentation source, and configuration files.
    h. "Object" form shall mean any form resulting from mechanical transformation or
       translation of a Source form, including but not limited to compiled object code,
       generated documentation, and conversions to other media types.
    i. "Non-Commercial" shall mean for research or evaluation purposes only.

2. Grant of Rights
    a. You are granted a non-exclusive, worldwide, non-transferable and royalty-free
       limited license under our intellectual property or other rights owned by us
       embodied in the Materials to use, reproduce, distribute, copy, create derivative
       works of, and make modifications to the Materials FOR NON-COMMERCIAL PURPOSES ONLY.
    b. You shall not use the Materials for any commercial purpose without obtaining a
       separate commercial license from us. If you wish to use the Materials commercially,
       you shall request a license from us at model-business@notice.qwencloud.com.

3. Redistribution
Subject to Section 2 (Grant of Rights), you may distribute copies or make the Materials,
or derivative works thereof, available as part of a product or service that contains any
of them, with or without modifications, and in Source or Object form, provided that you
meet the following conditions:
    a. You shall give any other recipients of the Materials or derivative works a copy of
       this Agreement;
    b. You shall cause any modified files to carry prominent notices stating that you
       changed the files;
    c. You shall retain in all copies of the Materials that you distribute the following
       attribution notices within a "Notice" text file distributed as a part of such
       copies: "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c)
       2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved."; and
    d. You may add your own copyright statement to your modifications and may provide
       additional or different license terms and conditions for use, reproduction, or
       distribution of your modifications, or for any such derivative works as a whole,
       provided your use, reproduction, and distribution of the work otherwise complies
       with the terms and conditions of this Agreement.

4. Rules of use
    a. The Materials may be subject to export controls or restrictions in China, the
       United States or other countries or regions. You shall comply with applicable laws
       and regulations in your use of the Materials.
    b. If you use the Materials or any outputs or results therefrom to create, train,
       fine-tune, or improve an AI model that is distributed or made available, you shall
       prominently display "Built with Qwen" or "Improved using Qwen" in the related
       product documentation.
    c. You shall not use "Qwen" as the primary name or identifier of any derivative works
       or products; reasonable descriptive use (e.g., "fine-tuned from Qwen Image") is
       permitted.

5. Intellectual Property
    a. We retain ownership of all intellectual property rights in and to the Materials and
       derivatives made by or for us. Conditioned upon compliance with the terms and
       conditions of this Agreement, with respect to any derivative works and
       modifications of the Materials that are made by you, you are and will be the owner
       of such derivative works and modifications.
    b. No trademark license is granted to use the trade names, trademarks, service marks,
       or product names of us, except as required to fulfill notice requirements under
       this Agreement or as required for reasonable and customary use in describing and
       redistributing the Materials.
    c. If you commence a lawsuit or other proceedings (including a cross-claim or
       counterclaim in a lawsuit) against us or any entity alleging that the Materials or
       any output therefrom, or any part of the foregoing, infringe any intellectual
       property or other right owned or licensable by you, then all licenses granted to
       you under this Agreement shall terminate as of the date such lawsuit or other
       proceeding is commenced or brought.

6. Disclaimer of Warranty and Limitation of Liability
    a. We are not obligated to support, update, provide training for, or develop any
       further version of the Qwen Materials or to grant any license thereto.
    b. THE MATERIALS ARE PROVIDED "AS IS" WITHOUT ANY EXPRESS OR IMPLIED WARRANTY OF ANY
       KIND INCLUDING WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR FITNESS FOR A
       PARTICULAR PURPOSE. WE MAKE NO WARRANTY AND ASSUME NO RESPONSIBILITY FOR THE
       SAFETY OR STABILITY OF THE MATERIALS AND ANY OUTPUT THEREFROM.
    c. IN NO EVENT SHALL WE BE LIABLE TO YOU FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED
       TO ANY DIRECT, OR INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM YOUR USE
       OR INABILITY TO USE THE MATERIALS OR ANY OUTPUT OF IT, NO MATTER HOW IT'S CAUSED.
    d. You will defend, indemnify and hold harmless us from and against any claim by any
       third party arising out of or related to your use or distribution of the Materials.

7. Survival and Termination.
    a. The term of this Agreement shall commence upon your acceptance of this Agreement or
       access to the Materials and will continue in full force and effect until terminated
       in accordance with the terms and conditions herein.
    b. We may terminate this Agreement if you breach any of the terms or conditions of
       this Agreement. Upon termination of this Agreement, you must delete and cease use
       of the Materials. Sections 6 and 8 shall survive the termination of this Agreement.

8. Governing Law and Jurisdiction.
    a. This Agreement and any dispute arising out of or relating to it will be governed by
       the laws of China, without regard to conflict of law principles, and the UN
       Convention on Contracts for the International Sale of Goods does not apply to this
       Agreement.
    b. The People's Courts in Hangzhou City shall have exclusive jurisdiction over any
       dispute arising out of this Agreement.

9. Other Terms and Conditions.
    a. Any arrangements, understandings, or agreements regarding the Material not stated
       herein are separate from and independent of the terms and conditions of this
       Agreement. You shall request a separate license from us, if you use the Materials
       in ways not expressly agreed to in this Agreement.
    b. We shall not be bound by any additional or different terms or conditions
       communicated by you unless expressly agreed.
```

## 2. FLUX.3 Image Klein

来源：<https://huggingface.co/black-forest-labs/flux-3-image-klein>

许可证：**FLUX Kommunity License v1.0**（Copyright Black Forest Labs Inc.）。非商业使用免费；商业使用需为 Qualifying User（年营收 < 500 万美元）或另行取得商业许可（<https://bfl.ai/licensing>）。

按协议第 3(b) 条要求的声明 / Attribution Notice required by Section 3(b)：

> This FLUX Model is licensed by Black Forest Labs Inc. under the FLUX Kommunity License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

与本页相关的条款摘要：

- **第 1(a) 条**：Output 不构成 Derivative。本页展示的是 Output 与统计量，不是模型或其衍生物，因此不涉及第 3(a) 条"向第三方提供许可证副本"的分发义务。
- **第 2(d) 条**：BFL 不对 Output 主张所有权；Output 的后续使用由使用者负责。
- **第 2(e) 条**：对外展示 Output 前须经过对违法/侵权内容的审查，并在适用法律要求时标注内容由人工智能生成。
- **第 4(b) 条**：不得用 FLUX 的 Output 或结果去改进功能相似的其他人工智能模型。**本站未做任何此类用途。**

完整许可证随模型分发于 `flux-3-image-klein/LICENSE.md`，也可在 Hugging Face 仓库页面查看。

## 3. AI 生成内容声明 / AI-generated content disclosure

本页（以及对应的博客文章）中出现的所有图像，均由上述模型在 NVIDIA A100 上生成。它们是**合成图像，不是真实照片**；画面中的人物、动物、场景、文字标识均不指向现实中的任何实体。若与真实图像相似，纯属巧合。按 FLUX Kommunity License 第 2(e) 条与通行的 AI 内容披露惯例，在此明确标注。

## 4. 本目录代码

`index.html`、`style.css`、`js/*.js` 均为本博客编写，与上述模型无衍生关系。`tools/image-viz/` 下的脚本（`arch_probe.py`、`export_web.py`、`montage.py`）同样是本博客编写，运行时调用 `diffusers`（Apache-2.0）、`transformers`（Apache-2.0）、`torch`（BSD-3-Clause）与 `Pillow`（MIT-CMU）。

## 5. 姊妹篇

`static/llm-viz/` 是另一个项目，许可证见其目录下的 `LICENSE-THIRD-PARTY.md`。
