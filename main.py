from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import re

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

SYSTEM = """You are a 6G network infrastructure analyst 
and Web3 DePIN researcher with deep knowledge of both 
NVIDIA 6G technology and decentralized physical 
infrastructure networks. You are inside NVIDIA's 
6G Developer Program."""

PROMPT = """Score this DePIN project on its 6G network 
performance and readiness. Be specific and technical.

Project: {name}
Type: {type}
Description: {description}
Website: {website}

Return ONLY valid JSON no markdown no backticks:
{{
  "overall_score": <1-10 integer>,
  "bandwidth_dependency": {{
    "score": <1-10>,
    "explanation": "one sentence why"
  }},
  "latency_sensitivity": {{
    "score": <1-10>,
    "explanation": "one sentence why"
  }},
  "device_density_support": {{
    "score": <1-10>,
    "explanation": "one sentence why"
  }},
  "real_world_deployment": {{
    "score": <1-10>,
    "explanation": "one sentence why"
  }},
  "revenue_model_strength": {{
    "score": <1-10>,
    "explanation": "one sentence why"
  }},
  "six_g_upgrade_potential": {{
    "score": <1-10>,
    "explanation": "one sentence why"
  }},
  "top_strength": "one sentence",
  "biggest_risk": "one sentence",
  "six_g_readiness_summary": "2-3 sentences",
  "verdict": "one punchy sentence",
  "recommended_action": "WATCH, BUY_ATTENTION, or AVOID"
}}"""


class ScoreRequest(BaseModel):
    name: str
    type: str
    description: str
    website: str = ""


@app.post("/score-depin")
async def score_project(req: ScoreRequest):
    try:
        msg = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            max_tokens=1500,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": PROMPT.format(
                        name=req.name,
                        type=req.type,
                        description=req.description,
                        website=req.website
                    )
                }
            ]
        )
        raw = re.sub(
            r"```json|```", "",
            msg.choices[0].message.content
        ).strip()
        return {"success": True, "data": json.loads(raw)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
