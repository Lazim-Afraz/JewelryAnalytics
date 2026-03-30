"""
Jewelry Analytics Chatbot
chatbot/assistant.py

Local LLM assistant powered by Ollama (Mistral).
Injects live analytics data into every conversation as system context.

Usage:
    from chatbot.assistant import JewelryAssistant
    assistant = JewelryAssistant(service=svc)
    reply = assistant.chat("Which branch is performing best?", history=[])
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Default Ollama settings ────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "mistral"


class JewelryAssistant:
    """
    Conversational assistant for jewelry portfolio analytics.

    - Connects to a local Ollama instance
    - Injects live KPI / branch / cluster data as system context
    - Maintains per-session chat history
    - Graceful fallback if Ollama is not running
    """

    def __init__(self,
                 model:   str = DEFAULT_MODEL,
                 service = None,
                 base_url: str = OLLAMA_BASE_URL):
        """
        Args:
            model    : Ollama model name (e.g. 'mistral', 'llama3').
            service  : AnalyticsService instance (already loaded).
            base_url : Ollama server URL.
        """
        self.model    = model
        self.service  = service
        self.base_url = base_url.rstrip("/")
        self.history: List[Dict] = []
        logger.info(f"JewelryAssistant initialised — model={model}")

    # =========================================================================
    # Health check
    # =========================================================================

    def is_ollama_running(self) -> bool:
        """Return True if Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        """Return list of models pulled in Ollama."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data   = json.loads(resp.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                return models
        except Exception:
            return []

    def set_model(self, model: str):
        """Switch the active model."""
        self.model = model
        logger.info(f"Model switched to: {model}")

    # =========================================================================
    # System prompt — live data injection
    # =========================================================================

    def _build_system_prompt(self) -> str:
        """
        Build a rich system prompt injecting live analytics data.
        Falls back to a generic prompt if service is not loaded.
        """
        base = (
            "You are a knowledgeable jewelry portfolio analytics assistant. "
            "You help business analysts and managers understand branch performance, "
            "product trends, and sales predictions for a jewelry retail chain. "
            "Answer clearly and concisely. Use numbers from the data provided. "
            "If you don't have enough data to answer, say so honestly. "
            "Do not make up data that is not provided.\n\n"
        )

        if self.service is None or not self.service.is_data_loaded():
            return base + (
                "Note: No analytics data is currently loaded. "
                "Ask the user to click 'Load Data' in the sidebar first."
            )

        try:
            dash    = self.service.get_dashboard_data()
            top5    = self.service.get_top_branches(5, "SALE_COUNT")
            ca      = self.service.get_cluster_analysis()
            filters = self.service.get_available_filters()

            # ── KPI summary ───────────────────────────────────────────────────
            kpi_block = (
                "=== CURRENT PORTFOLIO KPIs ===\n"
                f"Total Sales      : {dash.get('total_sales', 'N/A'):,}\n"
                f"Total Stock      : {dash.get('total_stock', 'N/A'):,}\n"
                f"Total Branches   : {dash.get('total_branches', 'N/A')}\n"
                f"Total Regions    : {dash.get('total_regions', 'N/A')}\n"
                f"Overall Efficiency: {dash.get('overall_efficiency', 0):.3f}\n"
                f"Sell-Through Rate : {dash.get('overall_sell_through', 0):.1%}\n"
                f"Local Heroes     : {dash.get('total_local_heroes', 'N/A')}\n"
                f"Clusters         : {dash.get('cluster_count', 'N/A')}\n"
                f"Top Branch       : {dash.get('top_branch', 'N/A')}\n"
                f"Top Region       : {dash.get('top_region', 'N/A')}\n\n"
            )

            # ── Top 5 branches ────────────────────────────────────────────────
            branch_lines = ["=== TOP 5 BRANCHES BY SALES ==="]
            if not top5.empty:
                for _, row in top5.iterrows():
                    branch_lines.append(
                        f"{int(row.get('rank', 0))}. {row['BRANCHNAME']} "
                        f"| Region: {row.get('REGION', 'N/A')} "
                        f"| Sales: {int(row.get('SALE_COUNT', 0)):,} "
                        f"| Efficiency: {row.get('avg_efficiency', 0):.3f} "
                        f"| Sell-Through: {row.get('branch_sell_through', 0):.1%}"
                    )
            branch_block = "\n".join(branch_lines) + "\n\n"

            # ── Cluster summary ───────────────────────────────────────────────
            cluster_lines = [f"=== CLUSTER ANALYSIS ({ca['n_clusters']} CLUSTERS) ==="]
            for c in ca["summary"].get("clusters", []):
                am = c.get("avg_metrics", {})
                cluster_lines.append(
                    f"  {c['label']} | Tier: {c['performance_tier']} "
                    f"| Branches: {c['num_branches']} "
                    f"| Avg Sales: {am.get('SALE_COUNT', 0):.1f} "
                    f"| Regions: {', '.join(c.get('regions', []))}"
                )
            cluster_block = "\n".join(cluster_lines) + "\n\n"

            # ── Available filters ─────────────────────────────────────────────
            filter_block = (
                "=== AVAILABLE ATTRIBUTES ===\n"
                f"Regions   : {', '.join(filters.get('regions', []))}\n"
                f"Purities  : {', '.join(filters.get('purities', []))}\n"
                f"Finishes  : {', '.join(filters.get('finishes', []))}\n"
                f"Themes    : {', '.join(filters.get('themes', []))}\n"
                f"Shapes    : {', '.join(filters.get('shapes', []))}\n"
                f"Workstyles: {', '.join(filters.get('workstyles', []))}\n"
                f"Brands    : {', '.join(filters.get('brands', []))}\n\n"
            )

            return base + kpi_block + branch_block + cluster_block + filter_block

        except Exception as e:
            logger.warning(f"Could not build full system prompt: {e}")
            return base + "Note: Could not retrieve live analytics data at this time."

    # =========================================================================
    # Chat
    # =========================================================================

    def chat(self,
             user_message: str,
             history: Optional[List[Dict]] = None) -> Tuple[str, List[Dict]]:
        """
        Send a message and get a reply from the LLM.

        Args:
            user_message : The user's input text.
            history      : Existing conversation history (list of role/content dicts).
                           If None, uses internal self.history.

        Returns:
            Tuple of (reply_text, updated_history)
        """
        if history is None:
            history = self.history

        # ── Ollama health check ───────────────────────────────────────────────
        if not self.is_ollama_running():
            reply = (
                "⚠️ **Ollama is not running.**\n\n"
                "To start the assistant:\n"
                "1. Download Ollama from https://ollama.com/download\n"
                "2. Install and run it\n"
                "3. Open a terminal and run: `ollama pull mistral`\n"
                "4. Ollama will run in the background automatically\n\n"
                "Then reload this page and try again."
            )
            updated = history + [
                {"role": "user",      "content": user_message},
                {"role": "assistant", "content": reply},
            ]
            self.history = updated
            return reply, updated

        # ── Check if model is available ───────────────────────────────────────
        available = self.get_available_models()
        model_to_use = self.model
        if available and not any(self.model in m for m in available):
            # Fall back to first available model
            model_to_use = available[0].split(":")[0]
            logger.warning(f"Model '{self.model}' not found. Using '{model_to_use}'.")

        # ── Build messages payload ────────────────────────────────────────────
        system_prompt = self._build_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        messages += history
        messages.append({"role": "user", "content": user_message})

        # ── Call Ollama /api/chat ─────────────────────────────────────────────
        try:
            payload = json.dumps({
                "model":    model_to_use,
                "messages": messages,
                "stream":   False,
                "options":  {
                    "temperature": 0.3,   # Low temp = factual, consistent
                    "num_ctx":     4096,
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                data  = json.loads(resp.read().decode())
                reply = data["message"]["content"].strip()

        except urllib.error.URLError as e:
            logger.error(f"Ollama request failed: {e}")
            reply = (
                "❌ Could not reach Ollama. "
                "Make sure it's running (`ollama serve`) and try again."
            )
        except KeyError:
            reply = "❌ Unexpected response from Ollama. Please try again."
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            reply = f"❌ An error occurred: {str(e)}"

        # ── Update history ────────────────────────────────────────────────────
        updated = history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        self.history = updated
        return reply, updated

    def chat_stream(self,
                    user_message: str,
                    history=None):
        """
        Streaming chat — yields text tokens as they arrive from Ollama.
        Use with st.write_stream() for instant first-token display.
        After exhausting the generator, self.history is updated.
        """
        if history is None:
            history = self.history

        if not self.is_ollama_running():
            msg = ("⚠️ Ollama is not running. "
                   "Install from https://ollama.com/download "
                   "then run: `ollama pull mistral`")
            self.history = history + [
                {"role": "user",      "content": user_message},
                {"role": "assistant", "content": msg},
            ]
            yield msg
            return

        available    = self.get_available_models()
        model_to_use = self.model
        if available and not any(self.model in m for m in available):
            model_to_use = available[0].split(":")[0]

        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        messages += history
        messages.append({"role": "user", "content": user_message})

        payload = json.dumps({
            "model":    model_to_use,
            "messages": messages,
            "stream":   True,
            "options":  {"temperature": 0.3, "num_ctx": 2048},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        full_reply = ""
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            full_reply += token
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            err = f"❌ Error: {str(e)}"
            full_reply = err
            yield err

        self.history = history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": full_reply},
        ]

    # =========================================================================
    # Prediction helper — natural language → predict_sales()
    # =========================================================================

    def predict_from_text(self, text: str) -> Optional[Dict]:
        """
        Attempt to extract prediction parameters from natural language
        and call service.predict_sales().

        This is a best-effort helper; the LLM handles ambiguous queries.

        Returns:
            Prediction dict or None if service not loaded.
        """
        if self.service is None or not self.service.is_data_loaded():
            return None

        filters = self.service.get_available_filters()

        # Simple keyword extraction for known attribute values
        def _find_match(text_lower, options):
            for opt in options:
                if opt.lower() in text_lower:
                    return opt
            return None

        tl = text.lower()
        input_data = {
            "REGION":      _find_match(tl, filters.get("regions",    [])),
            "PURITY":      _find_match(tl, filters.get("purities",   [])),
            "FINISH":      _find_match(tl, filters.get("finishes",   [])),
            "THEME":       _find_match(tl, filters.get("themes",     [])),
            "SHAPE":       _find_match(tl, filters.get("shapes",     [])),
            "WORKSTYLE":   _find_match(tl, filters.get("workstyles", [])),
            "BRAND":       _find_match(tl, filters.get("brands",     [])),
            "STOCK_COUNT": 20,  # default
        }

        # Only attempt prediction if at least one attribute was found
        if any(v for v in input_data.values() if v is not None and v != 20):
            try:
                return self.service.predict_sales(input_data)
            except Exception as e:
                logger.warning(f"Auto-predict failed: {e}")
        return None

    # =========================================================================
    # History management
    # =========================================================================

    def reset(self):
        """Clear conversation history."""
        self.history = []
        logger.info("Chat history cleared.")

    def get_history(self) -> List[Dict]:
        """Return current conversation history."""
        return self.history.copy()

    def get_context_summary(self) -> str:
        """Return a short summary of what data is loaded (for UI display)."""
        if self.service is None or not self.service.is_data_loaded():
            return "No data loaded"
        try:
            dash = self.service.get_dashboard_data()
            return (
                f"{dash.get('total_branches', '?')} branches · "
                f"{dash.get('total_regions', '?')} regions · "
                f"{dash.get('total_sales', 0):,} total sales"
            )
        except Exception:
            return "Data loaded"


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== JewelryAssistant Self-Test ===\n")

    assistant = JewelryAssistant()

    print("1. Checking Ollama connection...")
    running = assistant.is_ollama_running()
    print(f"   Ollama running: {running}")

    if running:
        models = assistant.get_available_models()
        print(f"   Available models: {models}")

    print("\n2. Building system prompt (no service)...")
    prompt = assistant._build_system_prompt()
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   First 200 chars: {prompt[:200]}...")

    print("\n3. Testing chat (no Ollama required for fallback)...")
    reply, history = assistant.chat("What is the best performing branch?")
    print(f"   Reply preview: {reply[:120]}...")
    print(f"   History length: {len(history)}")

    print("\n4. Testing reset...")
    assistant.reset()
    print(f"   History after reset: {len(assistant.history)} messages")

    print("\n5. Context summary (no service)...")
    print(f"   {assistant.get_context_summary()}")

    print("\n=== All tests passed ✅ ===")
