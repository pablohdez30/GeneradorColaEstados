"""
sniper_main.py — Bot Francotirador de alertas BTC/USDT.

Vigila el mercado 24/7 en múltiples timeframes.
Solo alerta cuando hay una oportunidad de ALTA CONFIANZA.
No ejecuta trades — solo envía alertas por Telegram.

Uso:
    python3 sniper_main.py
"""

import time
import signal as sig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sniper_bot.config import (
    SYMBOL, TIMEFRAMES, CHECK_INTERVAL_SECONDS,
    ALERT_COOLDOWN_MINUTES, MIN_SCORE_TO_ALERT,
    TELEGRAM_BOT_TOKEN,
)
from sniper_bot.analyzer import MarketAnalyzer
from sniper_bot.scorer import SignalScorer
from sniper_bot.notifier import TelegramNotifier
from sniper_bot.logger import setup_logger, AlertLogger

logger = setup_logger("sniper_main")
running = True


def signal_handler(signum, frame):
    global running
    running = False
    logger.info("Cerrando Sniper Bot...")


def main():
    global running

    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           🎯 SNIPER BOT — BTC/USDT                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Modo:        Solo ALERTAS (no ejecuta trades)              ║
    ║  Par:         BTC/USDT Perpetual                            ║
    ║  Timeframes:  15m, 1h, 4h (multi-timeframe)                ║
    ║  Mín. score:  4/7 para alertar                              ║
    ║  Cooldown:    1 hora entre alertas                           ║
    ║  Notifica:    Telegram + consola                            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Condiciones evaluadas:                                     ║
    ║  • ADX > 30 (tendencia fuerte)                              ║
    ║  • EMA 9/21 cross (dirección)                               ║
    ║  • RSI en zona favorable                                    ║
    ║  • Volumen > 2x media (institucionales)                     ║
    ║  • Fear & Greed extremo                                     ║
    ║  • Multi-timeframe alineados                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Inicializar
    analyzer = MarketAnalyzer()
    scorer = SignalScorer()
    notifier = TelegramNotifier()
    alert_logger = AlertLogger()

    last_alert_time = 0
    loop_count = 0
    last_daily_summary = 0

    # Mensaje de inicio
    notifier.send_startup_message()
    logger.info(f"Sniper Bot iniciado | Vigilando {SYMBOL} en {TIMEFRAMES}")
    logger.info(f"Telegram: {'configurado' if notifier.enabled else 'NO configurado'}")
    logger.info("Esperando oportunidades...")

    # ── Loop principal ─────────────────────────────────────────
    while running:
        try:
            loop_count += 1

            # Analizar mercado en todos los timeframes
            analysis = analyzer.analyze_all()

            if not analysis:
                logger.warning("No se pudieron obtener datos. Reintentando...")
                time.sleep(30)
                continue

            # Mostrar status cada 10 loops (~10 min)
            primary = analysis.get("1h", analysis.get("15m", {}))
            if loop_count % 10 == 0 and primary:
                price = primary.get("price", 0)
                adx = primary.get("adx", 0)
                rsi = primary.get("rsi", 50)
                vol = primary.get("volume_ratio", 1.0)
                fg = scorer.get_fear_greed()
                logger.info(
                    f"Status #{loop_count} | BTC: ${price:,.2f} | "
                    f"ADX: {adx:.0f} | RSI: {rsi:.0f} | Vol: {vol:.1f}x | "
                    f"F&G: {fg} | Alertas hoy: {alert_logger.get_alerts_today()}"
                )

            # Evaluar señal
            signal = scorer.evaluate(analysis)

            if signal:
                # Verificar cooldown
                elapsed = time.time() - last_alert_time
                cooldown_secs = ALERT_COOLDOWN_MINUTES * 60

                if elapsed >= cooldown_secs:
                    logger.info(
                        f"🎯 SEÑAL DETECTADA: {signal['direction']} | "
                        f"Score: {signal['score']}/7 | "
                        f"Confianza: {signal['confidence']:.0f}%"
                    )

                    # Enviar alerta
                    telegram_ok = notifier.send(signal)
                    alert_logger.log_alert(signal, telegram_ok)
                    last_alert_time = time.time()
                else:
                    remaining = int((cooldown_secs - elapsed) / 60)
                    logger.debug(
                        f"Señal detectada pero en cooldown ({remaining}min restantes)"
                    )

            # Resumen diario (cada 24h)
            if time.time() - last_daily_summary > 86400 and primary:
                fg = scorer.get_fear_greed()
                notifier.send_daily_summary(
                    alert_logger.get_alerts_today(),
                    primary.get("price", 0),
                    primary.get("adx", 0),
                    primary.get("rsi", 50),
                    fg,
                )
                last_daily_summary = time.time()

            # Esperar
            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            running = False
        except Exception as e:
            logger.error(f"Error en loop: {e}", exc_info=True)
            time.sleep(60)

    logger.info("Sniper Bot detenido.")


if __name__ == "__main__":
    main()
