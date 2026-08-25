from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from app.agent import SupportAgent
from app.trace import TraceLogger

app = typer.Typer(help="Aster & Row Customer Support Agent CLI")
console = Console()


@app.command()
def chat(
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable structured execution trace logging")
):
    """Starts an interactive customer support chat session."""
    console.print("\n[bold cyan]=======================================================[/bold cyan]")
    console.print("[bold white] Aster & Row Customer Support AI Agent [/bold white]")
    console.print("[dim]Type your message below. Type 'exit' or 'quit' to end.[/dim]")
    console.print("[bold cyan]=======================================================[/bold cyan]\n")

    agent = SupportAgent()
    session_id = "cli_session_1"

    while True:
        try:
            user_input = Prompt.ask("[bold green]Customer[/bold green]")
            if not user_input or user_input.strip().lower() in ("exit", "quit", "q"):
                console.print("\n[yellow]Thank you for contacting Aster & Row. Goodbye![/yellow]\n")
                break

            response = agent.process_message(user_input, session_id=session_id)

            # Format Assistant Output
            ans_panel = Panel(
                response.answer,
                title="[bold blue]Aster & Row Support[/bold blue]",
                border_style="blue"
            )
            console.print(ans_panel)

            # Display Citations if present
            if response.sources:
                sources_str = ", ".join([f"[cyan]{s}[/cyan]" for s in response.sources])
                console.print(f"[dim]Sources Cited:[/dim] {sources_str}")

            # Display Handoff Alert if recommended
            if response.handoff_recommended:
                console.print(f"[bold red][!] Human Handoff Recommended:[/bold red] [italic]{response.handoff_reason or 'Support assistance required'}[/italic]")


            # Display Debug Trace if enabled
            if debug and response.trace:
                trace_str = TraceLogger.format_trace_for_display(response.trace)
                console.print(Panel(trace_str, title="[yellow]Execution Trace (Debug)[/yellow]", border_style="yellow"))

            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended.[/yellow]\n")
            break


if __name__ == "__main__":
    app()
