"""
wandr.ai — AI Travel Planner
Single-file deployment: Python backend + entire frontend HTML embedded.
No build step. No npm. No separate frontend folder.
Deploy to Render.com in 2 minutes.
"""
import os, json, re
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Prompts ────────────────────────────────────────────────────────────────────
INTENT_SYS = """Extract travel intent from user input. Return ONLY valid JSON, no markdown, no explanation:
{"destination":"city, country","duration_days":3,"total_budget":100000,"currency":"INR","num_travellers":4,"interest_type":"sightseeing","accommodation_pref":"mid-range","transport_pref":"flight","group_type":"friends"}
Convert foreign currencies to INR (1 USD=83, 1 EUR=90 INR). Default budget 15000 INR if not mentioned."""

PLAN_SYS = """You are an expert world travel planner. Return ONLY valid JSON (no markdown fences, no explanation text before or after):
{
  "trip_title":"Catchy trip title",
  "summary":"2 exciting sentences about this trip",
  "weather":{"condition":"Sunny","avg_temp_c":22,"rain_prob":15,"icon":"☀️","packing_note":"Light layers recommended","alert":null},
  "transport":[
    {"id":"t1","mode":"flight","name":"Air India Non-stop","cost":45000,"departure":"10:00","arrival":"16:30","duration":"8h 30m","provider":"Air India","booking_url":"https://www.airindia.com","badge":"Recommended"},
    {"id":"t2","mode":"flight","name":"Emirates via Dubai","cost":38000,"departure":"02:00","arrival":"14:00","duration":"11h","provider":"Emirates","booking_url":"https://www.emirates.com","badge":"Best Value"},
    {"id":"t3","mode":"flight","name":"Budget Connecting","cost":30000,"departure":"06:00","arrival":"22:00","duration":"16h","provider":"IndiGo","booking_url":"https://www.goindigo.in","badge":"Budget"}
  ],
  "hotels":[
    {"id":"h1","name":"Hotel Name","stars":4,"location":"Central Area, City","cost_night":6000,"amenities":["WiFi","Breakfast","AC"],"map_query":"Hotel Name City Country","booking_url":"https://www.booking.com"},
    {"id":"h2","name":"Budget Hotel","stars":3,"location":"Old Town, City","cost_night":3200,"amenities":["WiFi","AC"],"map_query":"Budget Hotel City Country","booking_url":"https://www.agoda.com"},
    {"id":"h3","name":"Luxury Hotel","stars":5,"location":"Prime District, City","cost_night":14000,"amenities":["WiFi","Pool","Spa","Breakfast"],"map_query":"Luxury Hotel City Country","booking_url":"https://www.expedia.com"}
  ],
  "budget_breakdown":{"transport":45000,"hotel":24000,"food":15000,"activities":10000,"shopping":3000,"buffer":3000},
  "itinerary":[
    {"day":1,"theme":"Arrival & First Impressions","weather_note":"Clear and pleasant","activities":[
      {"time":"14:00","name":"Eiffel Tower","location":"Champ de Mars, Paris","duration_hours":2,"cost":500,"map_query":"Eiffel Tower Paris France","tip":"Book skip-the-line tickets online in advance"},
      {"time":"17:00","name":"Seine River Walk","location":"Pont Neuf, Paris","duration_hours":1.5,"cost":0,"map_query":"Pont Neuf Paris","tip":"Best at golden hour for photos"},
      {"time":"20:00","name":"Dinner at Le Marais","location":"Le Marais District, Paris","duration_hours":2,"cost":1500,"map_query":"Le Marais restaurants Paris","tip":"Try French onion soup and crème brûlée"}
    ]}
  ],
  "packing_list":{
    "clothing":["Light cotton shirts x4","Comfortable walking shoes","Formal outfit x1","Light jacket","Underwear & socks x5"],
    "toiletries":["Sunscreen SPF50","Lip balm","Travel toiletry kit","Hand sanitizer"],
    "documents":["Passport (valid 6+ months)","Travel insurance","Hotel bookings","Flight tickets","Visa printout"],
    "electronics":["Universal power adapter","Power bank 20000mAh","Smartphone + charger","Camera"],
    "medicines":["Pain relief tablets","ORS sachets","Antacids","Motion sickness pills"],
    "destination_specific":["Comfortable sandals","Reusable water bottle","Offline Google Maps downloaded"]
  },
  "events":["Local festival or event if applicable"],
  "booking_tips":["Book flights 2-3 months ahead","Use Booking.com for free cancellation hotels","Get travel insurance before departure","Download offline Google Maps"]
}
RULES: budget_breakdown MUST sum to total_budget. Generate ALL days of itinerary (not just day 1) with 3-4 real activities each. Use real attraction names for the exact destination."""

def parse_json(text: str) -> dict:
    text = re.sub(r'```json\s*', '', text.strip())
    text = re.sub(r'```\s*', '', text).strip()
    start = next((i for i, c in enumerate(text) if c in '{['), 0)
    try:
        return json.loads(text[start:])
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(m.group()) if m else {}

# ── API ────────────────────────────────────────────────────────────────────────
class PlanRequest(BaseModel):
    message: str

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "wandr.ai"}

@app.post("/api/plan")
async def plan(req: PlanRequest):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(500, "ANTHROPIC_API_KEY environment variable not set")
    client = anthropic.Anthropic(api_key=key)

    intent_msg = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=400,
        system=INTENT_SYS,
        messages=[{"role": "user", "content": req.message}]
    )
    intent = parse_json(intent_msg.content[0].text)

    plan_msg = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=4000,
        system=PLAN_SYS,
        messages=[{"role": "user", "content":
            f'User: "{req.message}"\nIntent: {json.dumps(intent)}\n'
            f'Generate ALL {intent.get("duration_days",3)} days. Budget: {intent.get("total_budget",15000)} INR. '
            f'Destination: {intent.get("destination","destination")}. Use real attraction names.'
        }]
    )
    plan_data = parse_json(plan_msg.content[0].text)
    return {"intent": intent, "plan": plan_data}

# ── Embedded HTML App ──────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>wandr.ai — AI Travel Planner</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,400&family=Outfit:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f0e0c;--bg2:#1a1814;--bg3:#242018;--card:#1e1c18;
  --brd:#2e2a22;--brd2:#3d3828;
  --gold:#d4a843;--goldL:#e8c060;--goldP:rgba(212,168,67,.13);
  --teal:#2a9d8f;--tealP:rgba(42,157,143,.13);
  --rose:#e76f51;--green:#52b788;
  --tx:#f0ead8;--tx2:#a09070;--tx3:#6a5f48;
}
html,body{min-height:100%;background:var(--bg);color:var(--tx);font-family:'Outfit',sans-serif;overflow-x:hidden}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:var(--bg2)}::-webkit-scrollbar-thumb{background:var(--brd2);border-radius:99px}
@keyframes slideUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-8px)}}

/* HEADER */
.hdr{position:sticky;top:0;z-index:99;height:56px;background:var(--bg);border-bottom:1px solid var(--brd);display:flex;align-items:center;justify-content:space-between;padding:0 20px}
.logo{font-family:'Cormorant Garamond',serif;font-size:22px;font-weight:700;color:var(--goldL)}
.logo span{color:var(--tx3);font-weight:300}
.hdr-badge{background:var(--goldP);border:1px solid rgba(212,168,67,.4);color:var(--gold);font-size:10px;font-weight:700;letter-spacing:.08em;padding:3px 10px;border-radius:99px;text-transform:uppercase}

/* LAYOUT */
.layout{display:flex;min-height:calc(100vh - 56px)}

/* SIDEBAR */
.sidebar{width:252px;flex-shrink:0;background:var(--bg2);border-right:1px solid var(--brd);padding:16px 12px;display:flex;flex-direction:column;gap:9px;overflow-y:auto}
@media(max-width:680px){.sidebar{display:none}}
.sb-ttl{font-family:'Cormorant Garamond',serif;font-size:13px;font-weight:700;color:var(--gold);border-bottom:1px solid var(--brd);padding-bottom:7px;margin-bottom:2px}
.ex{background:var(--bg3);border:1px solid var(--brd);border-radius:11px;padding:11px 12px;cursor:pointer;transition:all .2s}
.ex:hover{border-color:var(--gold);background:var(--goldP)}
.ex-icon{font-size:16px;margin-bottom:3px}
.ex-title{font-size:11px;font-weight:700;color:var(--gold);margin-bottom:2px}
.ex-text{font-size:10px;color:var(--tx3);line-height:1.4}
.feat{display:flex;align-items:center;gap:7px;font-size:10px;color:var(--tx3)}
.feat-dot{width:4px;height:4px;border-radius:50%;background:var(--gold);flex-shrink:0}

/* MAIN */
.main{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden}

/* HERO */
.hero{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:40px 20px;text-align:center}
.hero-badge{background:var(--tealP);border:1px solid rgba(42,157,143,.3);color:var(--teal);font-size:10px;font-weight:700;letter-spacing:.07em;padding:4px 13px;border-radius:99px;text-transform:uppercase;margin-bottom:20px}
.hero-h{font-family:'Cormorant Garamond',serif;font-size:clamp(1.9rem,5vw,3.2rem);font-weight:700;line-height:1.1;margin-bottom:14px;max-width:560px}
.hero-h em{color:var(--goldL);font-style:italic}
.hero-sub{font-size:14px;color:var(--tx2);max-width:420px;line-height:1.7;margin-bottom:28px}
.hero-chips{display:flex;gap:9px;flex-wrap:wrap;justify-content:center}
.chip{padding:7px 15px;background:var(--bg3);border:1px solid var(--brd2);border-radius:99px;font-size:12px;color:var(--tx2);cursor:pointer;transition:all .2s}
.chip:hover{border-color:var(--gold);color:var(--gold)}

/* CONVO */
.convo{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:18px}
.msg-u{align-self:flex-end;background:var(--bg3);border:1px solid var(--brd2);border-radius:18px 18px 4px 18px;padding:11px 16px;max-width:70%;font-size:14px;line-height:1.5;animation:slideUp .3s ease}
.msg-err{background:#2a1010;border:1px solid rgba(231,111,81,.3);border-radius:12px;padding:12px 16px;font-size:13px;color:var(--rose);animation:slideUp .3s}
.thinking{display:flex;align-items:center;gap:10px;color:var(--tx2);font-size:13px;font-style:italic;padding:4px 0}
.dots{display:flex;gap:5px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--gold);animation:bounce 1.2s infinite}

/* INPUT */
.input-area{padding:14px 20px 18px;border-top:1px solid var(--brd);background:var(--bg);flex-shrink:0}
.in-tabs{display:flex;gap:6px;margin-bottom:9px}
.in-tab{padding:4px 12px;border-radius:99px;font-size:11px;font-weight:700;border:1px solid var(--brd);background:transparent;color:var(--tx3);cursor:pointer;transition:all .15s;font-family:'Outfit',sans-serif}
.in-tab.on{background:var(--gold);color:var(--bg);border-color:var(--gold)}
.in-box{background:var(--bg2);border:1px solid var(--brd);border-radius:15px;padding:13px;transition:border-color .2s}
.in-box:focus-within{border-color:var(--gold)}
.in-row{display:flex;gap:9px;align-items:flex-end}
textarea{flex:1;resize:none;border:none;background:transparent;font-family:'Outfit',sans-serif;font-size:14px;line-height:1.5;color:var(--tx);min-height:46px;max-height:120px}
textarea:focus,input:focus,button:focus{outline:none}
.send{background:var(--gold);color:var(--bg);border:none;border-radius:10px;width:44px;height:44px;font-size:18px;cursor:pointer;transition:all .15s;flex-shrink:0}
.send:hover{background:var(--goldL);transform:scale(1.06)}
.send:disabled{background:var(--brd2);color:var(--tx3);cursor:not-allowed;transform:none}
.upload-zone{border:2px dashed var(--brd2);border-radius:11px;padding:26px;text-align:center;cursor:pointer;transition:all .2s}
.upload-zone:hover{border-color:var(--gold);background:var(--goldP)}
.url-row{display:flex;gap:8px}
.url-in{flex:1;padding:9px 13px;background:var(--bg3);border:1px solid var(--brd);border-radius:9px;font-family:'Outfit',sans-serif;font-size:13px;color:var(--tx)}
.url-btn{padding:9px 15px;background:var(--teal);color:#fff;border:none;border-radius:9px;font-weight:600;font-size:13px;cursor:pointer;font-family:'Outfit',sans-serif}
.hint{text-align:center;font-size:10px;color:var(--tx3);margin-top:7px}

/* TRIP CARD */
.trip-card{border:1px solid var(--brd2);border-radius:18px;overflow:hidden;animation:slideUp .4s ease}
.plan-hdr{background:linear-gradient(135deg,#1a1208,#0d1a14,#1a0d08);padding:22px;position:relative;overflow:hidden}
.plan-hdr::before{content:'';position:absolute;top:-60px;right:-60px;width:180px;height:180px;border-radius:50%;background:rgba(212,168,67,.07)}
.plan-title{font-family:'Cormorant Garamond',serif;font-size:24px;font-weight:700;color:var(--goldL);position:relative;margin-bottom:5px;line-height:1.2}
.plan-sum{font-size:12px;color:var(--tx2);font-style:italic;margin-bottom:14px;position:relative;line-height:1.5}
.plan-meta{display:flex;flex-wrap:wrap;gap:12px;position:relative}
.pm{display:flex;align-items:center;gap:4px;font-size:11px;color:var(--tx2)}
.pm strong{color:var(--goldL)}
.plan-tabs{display:flex;overflow-x:auto;background:var(--bg2);border-bottom:1px solid var(--brd);scrollbar-width:none}
.plan-tabs::-webkit-scrollbar{display:none}
.ptab{padding:11px 14px;white-space:nowrap;font-size:11px;font-weight:600;border:none;background:transparent;cursor:pointer;color:var(--tx3);border-bottom:3px solid transparent;font-family:'Outfit',sans-serif;transition:all .15s}
.ptab.on{color:var(--goldL);border-bottom-color:var(--gold)}
.ptab:hover:not(.on){background:rgba(255,255,255,.03)}
.pane{padding:18px;background:var(--bg)}

/* WEATHER */
.wx{display:flex;align-items:center;gap:11px;background:var(--tealP);border:1px solid rgba(42,157,143,.25);border-radius:11px;padding:10px 15px;margin-bottom:14px}
.wx-main{font-weight:600;color:var(--teal);font-size:12px}
.wx-sub{font-size:10px;color:var(--tx2);margin-top:2px}
.wx-alert{margin-left:auto;background:#3a2a10;border:1px solid rgba(212,168,67,.5);border-radius:7px;padding:3px 8px;font-size:10px;color:var(--gold);white-space:nowrap}

/* BUDGET */
.bdg-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:9px;margin-bottom:12px}
.bdg-card{background:var(--bg3);border:1px solid var(--brd);border-radius:11px;padding:12px 9px;text-align:center}
.bdg-lbl{font-size:8px;text-transform:uppercase;letter-spacing:.07em;color:var(--tx3);font-weight:600;margin-top:3px}
.bdg-amt{font-family:'DM Mono',monospace;font-size:13px;font-weight:600;color:var(--tx);margin:3px 0}
.bdg-pct{font-size:10px;color:var(--gold)}
.bdg-bar{height:3px;background:var(--brd);border-radius:99px;overflow:hidden;margin-top:7px}
.bdg-fill{height:100%;border-radius:99px}
.bdg-total{display:flex;justify-content:space-between;align-items:center;background:var(--bg2);border:1px solid rgba(212,168,67,.3);border-radius:11px;padding:13px 16px;font-weight:700;font-size:14px}
.bdg-total-amt{font-family:'DM Mono',monospace;font-size:18px;color:var(--goldL)}

/* TRANSPORT */
.tr-list{display:flex;flex-direction:column;gap:9px}
.tr-card{display:flex;align-items:center;gap:11px;border:2px solid var(--brd);border-radius:12px;padding:13px 14px;cursor:pointer;transition:all .2s}
.tr-card.sel{border-color:var(--gold);background:var(--goldP)}
.tr-card:hover:not(.sel){border-color:var(--brd2)}
.tr-name{font-weight:600;font-size:13px;color:var(--tx)}
.tr-meta{font-size:11px;color:var(--tx2);margin-top:2px}
.tr-price{font-family:'DM Mono',monospace;color:var(--goldL);font-weight:700;font-size:14px;text-align:right}
.tr-book{font-size:10px;color:var(--teal);text-decoration:none;display:block;text-align:right;margin-top:2px}
.tr-badge{font-size:9px;font-weight:700;letter-spacing:.05em;padding:2px 7px;border-radius:99px;text-transform:uppercase;white-space:nowrap}

/* HOTELS */
.ht-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(195px,1fr));gap:9px}
.ht-card{border:2px solid var(--brd);border-radius:12px;padding:13px;cursor:pointer;transition:all .2s}
.ht-card.sel{border-color:var(--gold);background:var(--goldP)}
.ht-card:hover:not(.sel){border-color:var(--brd2);transform:translateY(-2px)}
.ht-name{font-weight:700;font-size:13px;color:var(--tx);margin-bottom:3px}
.ht-stars{color:#f59e0b;font-size:11px;margin-bottom:3px}
.ht-loc{font-size:11px;color:var(--tx2);margin-bottom:7px}
.ht-tags{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:7px}
.ht-tag{font-size:10px;background:var(--bg2);padding:1px 6px;border-radius:99px;color:var(--tx2)}
.ht-price{font-family:'DM Mono',monospace;font-size:14px;font-weight:700;color:var(--goldL)}
.ht-price span{font-family:'Outfit',sans-serif;font-size:10px;color:var(--tx2);font-weight:400}
.ht-map{font-size:11px;color:var(--teal);display:block;margin-top:5px;text-decoration:none}

/* ITINERARY */
.itin{display:flex;flex-direction:column;gap:9px}
.day-card{border:1px solid var(--brd);border-radius:12px;overflow:hidden}
.day-hdr{display:flex;align-items:center;gap:9px;background:var(--bg2);padding:11px 13px;cursor:pointer;user-select:none}
.day-num{background:var(--gold);color:var(--bg);width:27px;height:27px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0}
.day-theme{flex:1;font-weight:600;font-size:13px}
.day-wx{font-size:10px;color:var(--tx3);margin-right:6px}
.day-acts{padding:9px;display:flex;flex-direction:column;gap:7px;background:var(--bg)}
.act{display:flex;gap:9px;align-items:flex-start;background:var(--bg3);border-radius:9px;padding:11px}
.act-time{font-family:'DM Mono',monospace;font-size:11px;color:var(--gold);font-weight:600;min-width:42px}
.act-name{font-weight:600;font-size:12px;color:var(--tx)}
.act-loc{font-size:10px;color:var(--tx2);margin-top:2px}
.act-tip{font-size:10px;color:var(--teal);font-style:italic;margin-top:3px}
.act-map{font-size:10px;color:var(--teal);text-decoration:none;display:inline-block;margin-top:3px}
.act-cost{font-family:'DM Mono',monospace;font-size:11px;color:var(--goldL);font-weight:600;white-space:nowrap}

/* PACKING */
.pk-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:9px}
.pk-cat{background:var(--bg3);border:1px solid var(--brd);border-radius:11px;padding:11px}
.pk-cat-title{font-weight:700;font-size:11px;color:var(--tx);margin-bottom:7px}
.pk-item{display:flex;align-items:center;gap:6px;font-size:11px;padding:3px 0;color:var(--tx2);cursor:pointer}
.pk-chk{width:14px;height:14px;border-radius:3px;border:1.5px solid var(--brd2);display:flex;align-items:center;justify-content:center;font-size:9px;color:#fff;flex-shrink:0;transition:all .15s}
.pk-done{background:var(--green)!important;border-color:var(--green)!important}
.pk-struck{text-decoration:line-through;color:var(--tx3)}

/* SPLIT */
.sp-inrow{display:flex;gap:7px;margin-bottom:12px}
.sp-in{flex:1;padding:8px 12px;background:var(--bg3);border:1px solid var(--brd);border-radius:8px;font-family:'Outfit',sans-serif;font-size:13px;color:var(--tx)}
.sp-add{padding:8px 14px;background:var(--teal);color:#fff;border:none;border-radius:8px;font-weight:600;font-size:13px;cursor:pointer;font-family:'Outfit',sans-serif}
.sp-person{display:flex;justify-content:space-between;align-items:center;background:var(--bg3);border-radius:8px;padding:10px 12px;margin-bottom:6px}
.sp-amt{font-family:'DM Mono',monospace;color:var(--goldL);font-weight:700}
.sp-res{padding:10px 12px;background:var(--tealP);border:1px solid rgba(42,157,143,.3);border-radius:8px;font-size:13px;color:var(--teal);font-weight:600;margin-top:10px}

/* ALERTS */
.al-panel{display:flex;gap:11px;align-items:flex-start;background:var(--bg3);border:1px solid var(--brd);border-radius:11px;padding:13px;margin-bottom:9px}
.al-title{font-weight:700;font-size:12px;color:var(--gold)}
.al-desc{font-size:11px;color:var(--tx2);margin-top:3px;line-height:1.5}
.al-btn{margin-top:7px;padding:4px 10px;background:transparent;border:1px solid var(--teal);border-radius:6px;color:var(--teal);cursor:pointer;font-size:11px;font-weight:700;font-family:'Outfit',sans-serif}
.al-btn:hover{background:var(--teal);color:#fff}

/* ACTIONS */
.act-row{display:flex;gap:7px;flex-wrap:wrap;margin-top:16px;padding-top:14px;border-top:1px solid var(--brd)}
.a-btn{display:flex;align-items:center;gap:4px;padding:7px 13px;border-radius:8px;font-size:11px;font-weight:700;cursor:pointer;font-family:'Outfit',sans-serif;transition:all .15s}
.a-primary{background:var(--gold);color:var(--bg);border:none}
.a-primary:hover{background:var(--goldL)}
.a-outline{background:transparent;color:var(--tx2);border:1px solid var(--brd2)}
.a-outline:hover{border-color:var(--gold);color:var(--gold)}
</style>
</head>
<body>
<header class="hdr">
  <div class="logo">wandr<span>.</span>ai</div>
  <div style="display:flex;align-items:center;gap:10px">
    <span class="hdr-badge">✨ AI-Powered</span>
    <span style="font-size:11px;color:var(--tx3)">Budget-Optimised Travel Agent</span>
  </div>
</header>

<div class="layout">
  <aside class="sidebar">
    <div class="sb-ttl">✈️ Quick Examples</div>
    <div class="ex" onclick="useEx(0)"><div class="ex-icon">🗼</div><div class="ex-title">Paris with Friends</div><div class="ex-text">3 day trip to Paris with 4 friends, ₹1,00,000</div></div>
    <div class="ex" onclick="useEx(1)"><div class="ex-icon">🏯</div><div class="ex-title">Tokyo Adventure</div><div class="ex-text">5 days Tokyo for 2 people, ₹1,50,000</div></div>
    <div class="ex" onclick="useEx(2)"><div class="ex-icon">🏝️</div><div class="ex-title">Bali Escape</div><div class="ex-text">7 days Bali for a couple, ₹80,000</div></div>
    <div class="ex" onclick="useEx(3)"><div class="ex-icon">🗽</div><div class="ex-title">New York Solo</div><div class="ex-text">5 days New York City solo, ₹1,20,000</div></div>
    <div class="ex" onclick="useEx(4)"><div class="ex-icon">🌿</div><div class="ex-title">Kerala Backwaters</div><div class="ex-text">4 days Kerala for 2 people, ₹30,000</div></div>
    <div class="ex" onclick="useEx(5)"><div class="ex-icon">🏛️</div><div class="ex-title">MP Heritage</div><div class="ex-text">3 days Madhya Pradesh heritage, ₹20,000</div></div>
    <div class="sb-ttl" style="margin-top:6px">Features</div>
    <div class="feat"><div class="feat-dot"></div>Full day-by-day itinerary</div>
    <div class="feat"><div class="feat-dot"></div>Budget breakdown & optimizer</div>
    <div class="feat"><div class="feat-dot"></div>Hotel & flight options</div>
    <div class="feat"><div class="feat-dot"></div>Weather-adjusted packing list</div>
    <div class="feat"><div class="feat-dot"></div>Group expense calculator</div>
    <div class="feat"><div class="feat-dot"></div>Google Maps for every spot</div>
    <div class="feat"><div class="feat-dot"></div>Booking tips & smart alerts</div>
  </aside>

  <main class="main">
    <div class="hero" id="hero">
      <div class="hero-badge">🌍 Runs 24/7 Online — No Install Needed</div>
      <h1 class="hero-h">Plan your perfect trip <em>in seconds</em></h1>
      <p class="hero-sub">Type any destination + budget. Get a complete itinerary, hotel picks, flights, packing list and budget breakdown instantly.</p>
      <div class="hero-chips">
        <div class="chip" onclick="useEx(0)">🗼 Paris</div>
        <div class="chip" onclick="useEx(1)">🏯 Tokyo</div>
        <div class="chip" onclick="useEx(2)">🏝️ Bali</div>
        <div class="chip" onclick="useEx(3)">🗽 New York</div>
      </div>
    </div>

    <div class="convo" id="convo" style="display:none"></div>

    <div class="input-area">
      <div class="in-tabs">
        <button class="in-tab on" id="t-text" onclick="setMode('text')">💬 Text</button>
        <button class="in-tab" id="t-image" onclick="setMode('image')">🖼️ Image</button>
        <button class="in-tab" id="t-url" onclick="setMode('url')">🔗 URL</button>
      </div>
      <div class="in-box">
        <div id="p-text">
          <div class="in-row">
            <textarea id="ta" rows="2" placeholder="e.g. 3 day trip to Paris with 4 friends, budget ₹1,00,000..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();go()}"></textarea>
            <button class="send" id="sendBtn" onclick="go()">➜</button>
          </div>
        </div>
        <div id="p-image" style="display:none">
          <div class="upload-zone" onclick="setMode('text');document.getElementById('ta').value='Plan a 3-day heritage trip to Rajasthan under ₹18,000 for 2 people';document.getElementById('ta').focus()">
            <div style="font-size:26px;margin-bottom:7px">📸</div>
            <div style="font-size:13px;color:var(--tx2);font-weight:600">Click to simulate screenshot upload</div>
            <div style="font-size:10px;color:var(--tx3);margin-top:3px">Hotel pages, Instagram posts, attraction screenshots</div>
          </div>
        </div>
        <div id="p-url" style="display:none">
          <div class="url-row">
            <input class="url-in" id="urlIn" placeholder="Paste hotel or attraction URL..." onkeydown="if(event.key==='Enter'){setMode('text');document.getElementById('ta').value='Plan a 5-day trip to that destination, ₹50,000 for 2 people';document.getElementById('ta').focus()}"/>
            <button class="url-btn" onclick="setMode('text');document.getElementById('ta').value='Plan a 5-day trip to that destination, ₹50,000 for 2 people';document.getElementById('ta').focus()">Extract</button>
          </div>
        </div>
      </div>
      <div class="hint">Powered by Claude AI · Plans every destination worldwide · 100% online</div>
    </div>
  </main>
</div>

<script>
const EX=['3 day trip to Paris with 4 friends, budget ₹1,00,000','5 days Tokyo for 2 people, ₹1,50,000 budget, love food and culture','7 days Bali for a couple, ₹80,000 budget, beaches and temples','5 days New York City solo traveller, ₹1,20,000 budget','4 days Kerala backwaters for 2 people, ₹30,000 budget','3 day Madhya Pradesh heritage trip for 2 people, ₹20,000 budget'];
const BI={transport:'🚄',hotel:'🏨',food:'🍽️',activities:'🎭',shopping:'🛍️',buffer:'💰'};
const BC={transport:'#2a9d8f',hotel:'#8b5cf6',food:'#e76f51',activities:'#52b788',shopping:'#d4a843',buffer:'#6a9bc2'};
const TI={flight:'✈️',train:'🚄',bus:'🚌',ferry:'⛴️'};
const PI={clothing:'👕',toiletries:'🧴',documents:'📄',electronics:'🔌',medicines:'💊',destination_specific:'📍'};
const BDGC={'Recommended':{bg:'#1a3a2a',c:'#52b788'},'Best Value':{bg:'#2a2a1a',c:'#d4a843'},'Budget':{bg:'#1a1a2a',c:'#90b4e8'},'Fastest':{bg:'#2a1a2a',c:'#c084fc'}};
let thinking=null,planData={};

function useEx(i){document.getElementById('ta').value=EX[i];setMode('text');document.getElementById('ta').focus()}
function setMode(m){['text','image','url'].forEach(t=>{document.getElementById('p-'+t).style.display=t===m?'block':'none';document.getElementById('t-'+t).classList.toggle('on',t===m)})}

function showConvo(){document.getElementById('hero').style.display='none';document.getElementById('convo').style.display='flex'}
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function add(html){const d=document.createElement('div');d.innerHTML=html;document.getElementById('convo').appendChild(d.firstElementChild);scroll()}
function scroll(){setTimeout(()=>{const c=document.getElementById('convo');if(c)c.scrollTop=c.scrollHeight},80)}

function showThink(msg){
  if(thinking)thinking.remove();
  const d=document.createElement('div');
  d.className='thinking';
  d.innerHTML=`<div class="dots"><div class="dot" style="animation-delay:0s"></div><div class="dot" style="animation-delay:.2s"></div><div class="dot" style="animation-delay:.4s"></div></div><span>${esc(msg)}</span>`;
  document.getElementById('convo').appendChild(d);thinking=d;scroll();
}
function hideThink(){if(thinking){thinking.remove();thinking=null}}

async function go(){
  const txt=document.getElementById('ta').value.trim();
  if(!txt)return;
  document.getElementById('ta').value='';
  document.getElementById('sendBtn').disabled=true;
  showConvo();
  add(`<div class="msg-u">${esc(txt)}</div>`);
  showThink('🧠 Extracting travel intent...');
  try{
    showThink('🗺️ Building your complete trip plan — this takes ~10 seconds...');
    const r=await fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:txt})});
    if(!r.ok){const e=await r.json().catch(()=>({}));throw new Error(e.detail||'Server error '+r.status)}
    const{intent,plan}=await r.json();
    hideThink();
    renderCard(plan,intent);
  }catch(e){
    hideThink();
    add(`<div class="msg-err">❌ ${esc(e.message)}. Please try again.</div>`);
  }finally{
    document.getElementById('sendBtn').disabled=false;
  }
}

function renderCard(plan,intent){
  const uid='c'+Date.now();
  const total=intent.total_budget||15000;
  planData[uid]={plan,intent};
  const TABS=[['itin','📅 Itinerary'],['budget','💰 Budget'],['transport','✈️ Transport'],['hotels','🏨 Hotels'],['packing','🎒 Packing'],['split','👥 Group Split'],['alerts','🔔 Alerts']];
  const wx=plan.weather||{};

  const card=document.createElement('div');
  card.className='trip-card';
  card.innerHTML=`
  <div class="plan-hdr">
    <div class="plan-title">✈️ ${esc(plan.trip_title||'Your Trip Plan')}</div>
    <div class="plan-sum">${esc(plan.summary||'')}</div>
    <div class="plan-meta">
      <div class="pm">📍 <strong>${esc(intent.destination||'')}</strong></div>
      <div class="pm">📅 <strong>${intent.duration_days||3} days</strong></div>
      <div class="pm">👥 <strong>${intent.num_travellers||1} traveller${(intent.num_travellers||1)>1?'s':''}</strong></div>
      <div class="pm">💰 <strong>₹${Number(total).toLocaleString('en-IN')}</strong></div>
      <div class="pm">🎯 <strong style="text-transform:capitalize">${esc(intent.interest_type||'travel')}</strong></div>
    </div>
  </div>
  <div class="plan-tabs" id="${uid}-tabs">
    ${TABS.map((t,i)=>`<button class="ptab${i===0?' on':''}" onclick="switchTab('${uid}','${t[0]}',this)">${t[1]}</button>`).join('')}
  </div>
  <div class="pane" id="${uid}-pane">
    ${wxBar(wx)}
    ${buildItinerary(plan.itinerary||[])}
  </div>
  <div class="pane" style="padding-top:0;padding-bottom:0">
    <div class="act-row">
      <button class="a-btn a-primary" onclick="dlPlan('${uid}')">⬇️ Download Plan</button>
      <button class="a-btn a-outline" onclick="switchTab('${uid}','split',null)">👥 Split Expenses</button>
      <button class="a-btn a-outline" onclick="switchTab('${uid}','alerts',null)">🔔 Tips & Alerts</button>
      <button class="a-btn a-outline" onclick="cpSum('${uid}')">📋 Copy</button>
    </div>
  </div>`;

  // Store pane builders for tab switching
  card._panes={
    itin:()=>wxBar(wx)+buildItinerary(plan.itinerary||[]),
    budget:()=>wxBar(wx)+buildBudget(plan.budget_breakdown||{},total),
    transport:()=>wxBar(wx)+buildTransport(plan.transport||[]),
    hotels:()=>wxBar(wx)+buildHotels(plan.hotels||[]),
    packing:()=>wxBar(wx)+buildPacking(plan.packing_list||{}),
    split:()=>buildSplit(uid,total),
    alerts:()=>buildAlerts(plan.events||[],plan.booking_tips||[]),
  };
  card._uid=uid;

  document.getElementById('convo').appendChild(card);
  scroll();
}

function switchTab(uid,tabId,btn){
  const card=document.querySelector('.trip-card[data-uid="'+uid+'"]')||findCard(uid);
  if(!card)return;
  card.querySelectorAll('.ptab').forEach(b=>b.classList.remove('on'));
  if(btn)btn.classList.add('on');
  else{const tabs=['itin','budget','transport','hotels','packing','split','alerts'];const i=tabs.indexOf(tabId);if(i>=0)card.querySelectorAll('.ptab')[i]?.classList.add('on')}
  document.getElementById(uid+'-pane').innerHTML=card._panes[tabId]?card._panes[tabId]():'';
  // Re-wire split members after render
  if(tabId==='split') wireSplit(uid, planData[uid]?.intent?.total_budget||15000);
}

function findCard(uid){
  const cards=document.querySelectorAll('.trip-card');
  for(const c of cards){if(c._uid===uid)return c;}return null;
}

// ── Pane builders ─────────────────────────────────────────────────────────────
function wxBar(wx){
  if(!wx||!wx.condition)return'';
  return`<div class="wx"><span style="font-size:20px">${wx.icon||'🌤️'}</span><div><div class="wx-main">${esc(wx.condition)} · ${wx.avg_temp_c}°C · ${wx.rain_prob}% rain</div><div class="wx-sub">${esc(wx.packing_note||'')}</div></div>${wx.alert?`<div class="wx-alert">⚠️ ${esc(wx.alert)}</div>`:''}</div>`;
}

function buildBudget(bd,total){
  return`<div class="bdg-grid">${Object.entries(bd).map(([k,v])=>{const p=total?Math.round(v/total*100):0;return`<div class="bdg-card"><div style="font-size:18px">${BI[k]||'📌'}</div><div class="bdg-lbl">${k}</div><div class="bdg-amt">₹${Number(v).toLocaleString('en-IN')}</div><div class="bdg-pct">${p}%</div><div class="bdg-bar"><div class="bdg-fill" style="width:${p}%;background:${BC[k]||'#888'}"></div></div></div>`;}).join('')}</div><div class="bdg-total"><span>Total Budget</span><span class="bdg-total-amt">₹${Number(total).toLocaleString('en-IN')}</span></div>`;
}

function buildTransport(opts){
  return`<div class="tr-list">${opts.map((t,i)=>{const bc=BDGC[t.badge]||{};return`<div class="tr-card${i===0?' sel':''}" onclick="selCard(this,'.tr-list','.tr-card')"><div style="font-size:20px;min-width:30px;text-align:center">${TI[t.mode]||'🚗'}</div><div style="flex:1"><div class="tr-name">${esc(t.name)}</div><div class="tr-meta">${t.departure}→${t.arrival} · ${t.duration} · ${esc(t.provider)}</div></div>${t.badge?`<span class="tr-badge" style="background:${bc.bg||'#333'};color:${bc.c||'#fff'}">${t.badge}</span>`:''}<div><div class="tr-price">₹${Number(t.cost).toLocaleString('en-IN')}</div><a class="tr-book" href="${t.booking_url}" target="_blank" onclick="event.stopPropagation()">Book →</a></div></div>`;}).join('')}</div>`;
}

function buildHotels(hotels){
  return`<div class="ht-grid">${hotels.map((h,i)=>`<div class="ht-card${i===0?' sel':''}" onclick="selCard(this,'.ht-grid','.ht-card')"><div class="ht-name">${esc(h.name)}</div><div class="ht-stars">${'★'.repeat(Math.floor(h.stars))}${'☆'.repeat(5-Math.floor(h.stars))} ${h.stars}</div><div class="ht-loc">📍 ${esc(h.location)}</div><div class="ht-tags">${(h.amenities||[]).slice(0,4).map(a=>`<span class="ht-tag">${esc(a)}</span>`).join('')}</div><div class="ht-price">₹${Number(h.cost_night).toLocaleString('en-IN')} <span>/ night</span></div><a class="ht-map" href="https://www.google.com/maps/search/${encodeURIComponent(h.map_query||h.name)}" target="_blank">🗺 View on Map</a></div>`).join('')}</div>`;
}

function buildItinerary(days){
  return`<div class="itin">${days.map((d,i)=>`<div class="day-card"><div class="day-hdr" onclick="toggleDay(this)"><div class="day-num">${d.day}</div><div class="day-theme">${esc(d.theme)}</div><div class="day-wx">${esc(d.weather_note||'')}</div><div style="color:var(--tx3);font-size:12px;transition:transform .2s" class="dchev">${i===0?'▴':'▾'}</div></div><div class="day-acts"${i===0?'':' style="display:none"'}>${(d.activities||[]).map(a=>`<div class="act"><div class="act-time">${a.time}</div><div style="flex:1"><div class="act-name">${esc(a.name)}</div><div class="act-loc">📍 ${esc(a.location)}</div>${a.tip?`<div class="act-tip">💡 ${esc(a.tip)}</div>`:''}<a class="act-map" href="https://www.google.com/maps/search/${encodeURIComponent(a.map_query||a.name)}" target="_blank">🗺 Map</a></div>${a.cost>0?`<div class="act-cost">₹${Number(a.cost).toLocaleString('en-IN')}</div>`:''}</div>`).join('')}</div></div>`).join('')}</div>`;
}

function buildPacking(list){
  return`<div class="pk-grid">${Object.entries(list).map(([cat,items])=>`<div class="pk-cat"><div class="pk-cat-title">${PI[cat]||'📦'} ${cat.replace('_',' ').replace(/\b\w/g,l=>l.toUpperCase())}</div>${(items||[]).map(item=>`<div class="pk-item" onclick="togglePk(this)"><div class="pk-chk">✓</div>${esc(item)}</div>`).join('')}</div>`).join('')}</div>`;
}

function buildSplit(uid,total){
  return`<div><div style="font-weight:600;font-size:13px;margin-bottom:12px">Splitting ₹${Number(total).toLocaleString('en-IN')} among <span id="${uid}-sc">1 person</span></div><div class="sp-inrow"><input class="sp-in" id="${uid}-si" placeholder="Add person's name..."/><button class="sp-add" id="${uid}-ab">+ Add</button></div><div id="${uid}-sl"><div class="sp-person"><span style="font-weight:600;font-size:13px">👤 You</span><span class="sp-amt">₹${Number(total).toLocaleString('en-IN')}</span></div></div><div id="${uid}-sr" style="display:none" class="sp-res"></div></div>`;
}

const splitState={};
function wireSplit(uid,total){
  if(!splitState[uid])splitState[uid]={members:['You'],total};
  const ab=document.getElementById(uid+'-ab');
  const si=document.getElementById(uid+'-si');
  if(ab)ab.onclick=()=>{const n=si.value.trim();if(n){splitState[uid].members.push(n);si.value='';renderSplit(uid)}};
  if(si)si.onkeydown=e=>{if(e.key==='Enter'){const n=si.value.trim();if(n){splitState[uid].members.push(n);si.value='';renderSplit(uid)}}};
}
function renderSplit(uid){
  const st=splitState[uid];if(!st)return;
  const per=Math.round(st.total/st.members.length);
  const sc=document.getElementById(uid+'-sc');if(sc)sc.textContent=st.members.length+(st.members.length===1?' person':' people');
  const sl=document.getElementById(uid+'-sl');if(sl)sl.innerHTML=st.members.map((m,i)=>`<div class="sp-person"><span style="font-weight:600;font-size:13px">👤 ${esc(m)}</span><span style="display:flex;align-items:center;gap:9px"><span class="sp-amt">₹${per.toLocaleString('en-IN')}</span>${i>0?`<span style="color:var(--rose);cursor:pointer" onclick="rmSplit('${uid}',${i})">✕</span>`:''}</span></div>`).join('');
  const sr=document.getElementById(uid+'-sr');if(sr){if(st.members.length>1){sr.style.display='block';sr.textContent=`✅ Each person pays ₹${per.toLocaleString('en-IN')}`}else sr.style.display='none'}
}
function rmSplit(uid,idx){if(splitState[uid]){splitState[uid].members.splice(idx,1);renderSplit(uid)}}

function buildAlerts(events,tips){
  const ev=(events||[]).filter(Boolean);
  return(ev.length?`<div class="al-panel"><div style="font-size:18px">🎉</div><div><div class="al-title">Events During Your Trip</div><div class="al-desc">${ev.map(esc).join(' · ')}</div></div></div>`:'')+
  `<div class="al-panel"><div style="font-size:18px">🌦️</div><div><div class="al-title">Smart Weather Re-Planning</div><div class="al-desc">Weather changed? Click below to regenerate a weather-optimised itinerary instantly.</div><button class="al-btn" onclick="alert('Re-plan: This triggers a new AI call with updated weather forecast in production!')">🔄 Replan for Weather</button></div></div>`+
  (tips||[]).map((t,i)=>`<div class="al-panel"><div style="font-size:18px">💡</div><div><div class="al-title">Booking Tip ${i+1}</div><div class="al-desc">${esc(t)}</div></div></div>`).join('');
}

// ── Interactions ──────────────────────────────────────────────────────────────
function toggleDay(hdr){
  const acts=hdr.nextElementSibling,chev=hdr.querySelector('.dchev'),open=acts.style.display!=='none';
  acts.style.display=open?'none':'flex';chev.textContent=open?'▾':'▴';
}
function togglePk(item){
  const chk=item.querySelector('.pk-chk');chk.classList.toggle('pk-done');item.classList.toggle('pk-struck');
}
function selCard(card,listSel,cardSel){
  card.closest(listSel).querySelectorAll(cardSel).forEach(c=>c.classList.remove('sel'));card.classList.add('sel');
}
function dlPlan(uid){
  const d=planData[uid];if(!d)return;
  const b=new Blob([JSON.stringify(d,null,2)],{type:'application/json'});
  Object.assign(document.createElement('a'),{href:URL.createObjectURL(b),download:`trip-${(d.intent.destination||'plan').toLowerCase().replace(/\s+/g,'-')}.json`}).click();
}
function cpSum(uid){
  const d=planData[uid];if(!d)return;
  const t=`✈️ ${d.plan.trip_title}\n📍 ${d.intent.destination} · ${d.intent.duration_days} days · ₹${Number(d.intent.total_budget).toLocaleString('en-IN')}\n\n${d.plan.summary}\n\nGenerated by wandr.ai`;
  navigator.clipboard?.writeText(t).then(()=>alert('Copied to clipboard!'));
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
@app.get("/{path:path}", response_class=HTMLResponse)
async def serve(path: str = ""):
    # Only serve HTML for non-API routes
    if path.startswith("api/"):
        raise HTTPException(404)
    return HTMLResponse(HTML)
