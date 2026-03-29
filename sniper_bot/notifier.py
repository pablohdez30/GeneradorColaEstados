"""
notifier.py — Envío de alertas por Telegram.

Formatea la señal en un mensaje claro y lo envía al chat configurado.
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone

from sniper_bot.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


class TelegramNotifier:
    """Envía alertas formateadas por Telegram."""

    def __init__(self):
        self.enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        if not self.enabled:
            print("⚠️  Telegram no configurado. Las alertas solo se mostrarán en consola.")

    def format_signal(self, signal: dict) -> str:
        """Formatea la señal como mensaje de Telegram."""
        direction = signal["direction"]
        emoji = "🟢" if direction == "LONG" else "🔴"
        score = signal["score"]
        stars = "⭐" * min(score, 5)

        reasons_text = "\n".join(f"  • {r}" for r in signal["reasons"])

        msg = f"""
{emoji} *SEÑAL {direction} — BTC/USDT* {stars}

*Confianza:* {signal['confidence']:.0f}% (score {score}/7)
*Precio actual:* ${signal['entry']:,.2f}

📍 *Entrada sugerida:* ${signal['entry']:,.2f}
🛑 *Stop Loss:* ${signal['stop_loss']:,.2f} (-{signal['sl_pct']}%)
🎯 *Take Profit:* ${signal['take_profit']:,.2f} (+{signal['tp_pct']}%)
⚖️ *Risk/Reward:* {signal['risk_reward']}
💪 *Apalancamiento sugerido:* x{signal['leverage']}

*Razones:*
{reasons_text}

📊 *Indicadores:*
  RSI: {signal['indicators']['rsi']} | ADX: {signal['indicators']['adx']}
  Volumen: {signal['indicators']['volume_ratio']}x | F&G: {signal['fear_greed']}

⚠️ *Riesgo máximo sugerido: 2% del capital*
_No es consejo financiero. Siempre usa stop loss._
"""
        return msg.strip()

    def send(self, signal: dict) -> bool:
        """Envía la alerta por Telegram."""
        message = self.format_signal(signal)

        # Siempre mostrar en consola
        print("\n" + "=" * 60)
        print(message.replace("*", "").replace("_", ""))
        print("=" * 60 + "\n")

        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }).encode("utf-8")

            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                if result.get("ok"):
                    print("✅ Alerta enviada por Telegram")
                    return True
                else:
                    print(f"❌ Error Telegram: {result}")
                    return False
        except Exception as e:
            print(f"❌ Error enviando Telegram: {e}")
            return False

    def send_startup_message(self):
        """Envía mensaje de inicio."""
        if not self.enabled:
            return

        msg = """
🤖 *Sniper Bot iniciado*

Vigilando BTC/USDT en 15m, 1h y 4h.
Te avisaré cuando encuentre una oportunidad de alta confianza.

_Puede pasar horas o días sin alertar. Eso es normal._
"""
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg.strip(),
                "parse_mode": "Markdown",
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    def send_daily_summary(self, alerts_today: int, price: float, adx: float,
                           rsi: float, fear_greed: int):
        """Envía resumen diario para saber que sigue vivo."""
        if not self.enabled:
            return

        msg = f"""
📋 *Resumen diario — Sniper Bot*

💰 BTC: ${price:,.2f}
📊 ADX: {adx:.0f} | RSI: {rsi:.0f} | F&G: {fear_greed}
🔔 Alertas hoy: {alerts_today}

{"_Sin oportunidades claras hoy. Paciencia._" if alerts_today == 0 else ""}
"""
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg.strip(),
                "parse_mode": "Markdown",
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass
