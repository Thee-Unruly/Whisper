"""
Agentic Multi-Turn Conversational RAG Router with Submodule Clarification.
Flow:
1. Receives chat conversation history and current user message.
2. Checks if the query references a specific enterprise module or if ambiguity exists.
3. If module is not specified or ambiguous, asks the user clarifying follow-up question with interactive module options.
4. Once the target module/context is identified (or explicitly global), performs semantic vector search in PostgreSQL.
5. Synthesizes a grounded, conversational answer citing exact recordings, timestamps, and modules.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
import httpx

import db
import pipeline

logger = logging.getLogger("signal.agent")

AGENT_ROUTER_PROMPT = """You are the intelligence coordinator for an enterprise audio knowledge base.
Your job is to examine the user's message and the prior conversation history.

The system indexes recordings across these 12 Enterprise Submodules:
- 01_credit: 01. Credit (Loans, appraisal, credit committee, limits)
- 02_credit_portal: 02. Credit Portal (Customer application portal, submissions)
- 02_finance: 02. Finance (General ledger, accounting periods, journal vouchers, budgets, charts of accounts)
- 03_e_recruitment: 03. E-Recruitment (Vacancies, applicant screening, interview scorecards)
- 04_procurement: 04. Procurement (Purchase orders, requisitions, vendor approvals)
- 05_e_procurement: 05. E-Procurement (Supplier portal, tender bidding, quotation requests)
- 06_treasury: 06. Treasury (Cash flow, bank reconciliation, investments, transfers)
- 07_grc: 07. GRC (Governance, risk matrix, audit trails, compliance)
- 08_hr: 08. HR (Employee files, leave management, onboarding, contracts)
- 09_payroll: 09. Payroll (Salary calculations, deductions, PAYE, payslips)
- 10_edms: 10. EDMS (Document archiving, file classification, access controls)
- 11_power_bi: 11. Power BI (Analytics dashboards, reports, DAX, KPI tracking)

Analyze if the user's question clearly refers to a specific module or if clarification is needed.

Respond ONLY with a valid JSON object in this exact schema:
{
  "needs_clarification": true | false,
  "clarification_message": "Follow up question asking which specific submodule they are referring to...",
  "suggested_modules": ["02_finance", "06_treasury", "01_credit"],
  "resolved_submodule": "02_finance" | "all" | null,
  "search_query": "Cleaned standalone semantic search query"
}
"""

AGENT_ANSWER_PROMPT = """You are an expert enterprise software guide and AI assistant.
Your goal is to provide clear, actionable, and structured instructions based on the knowledge base.

OUTPUT STRUCTURE RULES:
1. **Module & Navigation Path** (First line):
   - State the clean breadcrumb path:
     `📍 Module: [Module Name]`
     `🧭 Navigation: [Top Menu] → [Section / Page] → [Action / Card]`

2. **Sequential Action Steps**:
   - Provide a concise, numbered breakdown of actions:
     `Step 1: [Action title] - [Clear explanation of what to click, select, or fill]`
     `Step 2: [Action title] - [Clear explanation]`
     `Step 3: [Action title] - [Clear explanation]`

3. **Key Notes & Validation Rules** (If applicable):
   - Mention any mandatory fields, approval conditions, or prerequisites.

4. **Formatting Constraints**:
   - Do NOT include timestamp numbers (like 04:12 - 05:30) or audio recording file names in your text response.
   - Keep the tone confident, clean, and directly instructive for end-users.
"""

async def process_agent_turn(messages: List[Dict[str, str]], forced_submodule: Optional[str] = None) -> Dict[str, Any]:
    """
    Processes one conversational turn:
    - Analyzes intent and submodule scope.
    - If ambiguous, returns clarification question.
    - If user replied to a clarification (e.g. 'Searching in 02. Finance'), restores original question for vector search.
    - Runs vector retrieval and generates structured, navigation-rich procedural answer.
    """
    if not messages:
        return {"reply": "Hello! How can I assist you with the knowledge base today?", "type": "message"}

    latest_msg = messages[-1]["content"].strip()
    
    # 1. Detect if this turn is a follow-up answer to a clarification question
    search_query = latest_msg
    if len(messages) >= 3:
        # Check if 2 turns ago was the original user question
        prior_user_msg = messages[-3]["content"].strip()
        prior_asst_msg = messages[-2]["content"].strip().lower()
        if "which" in prior_asst_msg and ("submodule" in prior_asst_msg or "module" in prior_asst_msg):
            # The search query should be the original question, scoped to this chosen module
            search_query = prior_user_msg

    # 2. Determine resolved submodule
    if forced_submodule and forced_submodule != "auto":
        resolved_module = forced_submodule
        needs_clarification = False
    else:
        lower_msg = latest_msg.lower()
        matched_mod = None
        module_keywords = {
            "02_finance": ["finance", "accounting", "ledger", "fiscal", "journal", "invoice", "chart of account"],
            "01_credit": ["credit appraisal", "loan limit", "collateral", "credit committee", "borrower"],
            "02_credit_portal": ["credit portal", "loan application portal", "borrower portal"],
            "03_e_recruitment": ["recruitment", "vacancy", "applicant", "interview", "job posting"],
            "04_procurement": ["procurement", "purchase order", "po approval", "requisition", "vendor order"],
            "05_e_procurement": ["e-procurement", "supplier portal", "tender", "bidding"],
            "06_treasury": ["treasury", "cash flow", "bank reconciliation", "liquidity"],
            "07_grc": ["risk matrix", "compliance", "audit log", "governance", "grc"],
            "08_hr": ["employee file", "leave request", "onboarding", "hr policy", "contract"],
            "09_payroll": ["payroll", "salary", "paye", "payslip", "deduction"],
            "10_edms": ["edms", "document management", "archive", "file classification"],
            "11_power_bi": ["power bi", "powerbi", "dashboard", "dax", "kpi report"],
        }
        
        for mod_code, kws in module_keywords.items():
            if any(kw in lower_msg for kw in kws):
                matched_mod = mod_code
                break

        # Check if user message is brief or ambiguous (e.g. "how do I create a new record?", "explain approval process")
        is_generic_question = len(latest_msg.split()) < 8 and not matched_mod
        
        prior_asked_module = False
        if len(messages) >= 2 and messages[-2].get("role") == "assistant" and "which" in messages[-2].get("content", "").lower() and "module" in messages[-2].get("content", "").lower():
            prior_asked_module = True

        if is_generic_question and not prior_asked_module:
            return {
                "reply": "I'd love to walk you through the exact steps! Which enterprise submodule or domain are you referring to?",
                "type": "clarification",
                "options": [
                    {"code": "02_finance", "label": "02. Finance"},
                    {"code": "01_credit", "label": "01. Credit & Lending"},
                    {"code": "04_procurement", "label": "04. Procurement"},
                    {"code": "08_hr", "label": "08. HR & Payroll"},
                    {"code": "06_treasury", "label": "06. Treasury"},
                    {"code": "all", "label": "🌐 Search All Modules"}
                ]
            }

        resolved_module = matched_mod or "all"

    # 3. Perform Vector RAG retrieval on the resolved module
    results = pipeline.search_kb(query=search_query, top_k=7, submodule=resolved_module)
    
    if not results:
        mod_label = resolved_module.replace("_", " ").title() if resolved_module != "all" else "Global KB"
        return {
            "reply": f"I checked the **{mod_label}** recordings in the database, but couldn't find any speech segments matching *\"{search_query}\"*. Would you like me to broaden the search across all submodules?",
            "type": "not_found",
            "options": [{"code": "all", "label": "Search across All Submodules"}]
        }

    # 4. Format transcript excerpts
    snippets = []
    for idx, r in enumerate(results, 1):
        m_start = int(r["start_time"] // 60)
        s_start = int(r["start_time"] % 60)
        m_end = int(r["end_time"] // 60)
        s_end = int(r["end_time"] % 60)
        ts = f"{m_start:02d}:{s_start:02d} - {m_end:02d}:{s_end:02d}"
        mod_name = r.get("submodule_name") or r.get("submodule_code") or "General"
        snippets.append(f"--- Excerpt {idx} (Submodule: [{mod_name}] | File: {r['source_file']} | Time: {ts}) ---\n{r['text']}")

    context_block = "\n\n".join(snippets)

    # 5. Call LLM for structured procedural synthesis
    cfg = pipeline.get_llm_config()
    chat_prompt = (
        f"Transcript Context from Recordings:\n{context_block}\n\n"
        f"User Question / Goal:\n{search_query}\n\n"
        f"Provide the complete step-by-step procedure with navigation breadcrumbs (where to click), detailed actions, and timestamp citations."
    )

    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": AGENT_ANSWER_PROMPT},
            {"role": "user", "content": chat_prompt}
        ],
        "temperature": 0.2
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(cfg["url"], json=payload, headers=cfg["headers"], timeout=35.0)
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
            else:
                answer = f"Found relevant step in **{results[0].get('submodule_name', 'KB')}** ({results[0]['source_file']}):\n\n" + results[0]["text"]
    except Exception as e:
        logger.error(f"Agent LLM synthesis error: {e}")
        answer = f"Here is the relevant excerpt retrieved from your recordings:\n\n> {results[0]['text']}"

    return {
        "reply": answer,
        "type": "answer",
        "sources": results,
        "resolved_submodule": resolved_module
    }

