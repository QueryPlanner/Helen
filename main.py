import json
import os
import subprocess
import sys
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt

# Load environment variables from .env
load_dotenv()

# Initialize Typer and Rich Console
app = typer.Typer(help="An LLM-in-a-loop agent with bash tool access via OpenRouter.")
console = Console()

# Verify API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    console.print("[bold red]Error: OPENROUTER_API_KEY environment variable not found.[/bold red]")
    console.print("Please set the key in your .env file or as an environment variable.")
    sys.exit(1)

# Initialize OpenAI Client pointing to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "exec_bash",
            "description": "Execute a bash command on the local machine and return its standard output, standard error, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                },
                "required": ["command"],
            },
        }
    }
]

def run_bash(command: str) -> dict:
    """Executes a bash command and returns output, errors, and status."""
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "exit_code": res.returncode
        }
    except Exception as e:
        return {
            "error": str(e)
        }

def agent_loop(
    messages: List[dict], 
    model: str, 
    system_instruction: Optional[str] = None
) -> List[dict]:
    """Runs the LLM-in-a-loop, executing bash tools when requested by the model via OpenAI SDK."""
    
    # Prefix system instruction if provided
    current_messages = list(messages)
    if system_instruction and not any(m.get("role") == "system" for m in current_messages):
        current_messages.insert(0, {"role": "system", "content": system_instruction})
        
    while True:
        with console.status(f"[bold green]Agent is thinking (using {model})..."):
            try:
                # OpenRouter API call
                response = client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=tools,
                )
            except Exception as e:
                console.print(f"[bold red]OpenRouter API Error:[/bold red] {e}")
                sys.exit(1)
                
        message = response.choices[0].message
        tool_calls = message.tool_calls
        
        # Build assistant message dict to append
        assistant_message = {
            "role": "assistant",
            "content": message.content or ""
        }
        
        # If model has reasoning details (for supported reasoning models), preserve them
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            assistant_message["reasoning_details"] = message.reasoning_details
            
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in tool_calls
            ]
        
        current_messages.append(assistant_message)
        
        if tool_calls:
            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                call_id = tool_call.id
                
                if name == "exec_bash":
                    try:
                        args_dict = json.loads(arguments)
                        command = args_dict.get("command")
                    except Exception as e:
                        console.print(f"[bold red]Failed to parse arguments:[/bold red] {e}")
                        continue
                        
                    if not command:
                        continue
                        
                    # Log command execution
                    console.print(Panel(
                        Syntax(command, "bash", theme="monokai", line_numbers=True),
                        title="[bold yellow]Executing Bash Command[/bold yellow]",
                        border_style="yellow"
                    ))
                    
                    # Run command
                    result = run_bash(command)
                    
                    # Print outputs nicely
                    if result.get("stdout"):
                        console.print(f"[bold green]stdout:[/bold green]\n{result['stdout'].strip()}")
                    if result.get("stderr"):
                        console.print(f"[bold red]stderr:[/bold red]\n{result['stderr'].strip()}")
                    if "error" in result:
                        console.print(f"[bold red]Error:[/bold red] {result['error']}")
                    console.print(f"[bold blue]Exit Code:[/bold blue] {result.get('exit_code', 'N/A')}\n")
                    
                    # Formulate tool response
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": "exec_bash",
                        "content": json.dumps(result)
                    }
                    current_messages.append(tool_message)
            continue
        else:
            # Print final response content
            if message.content:
                console.print(Panel(
                    message.content.strip(),
                    title="[bold green]Agent Response[/bold green]",
                    border_style="green"
                ))
            break
            
    return current_messages

@app.command()
def run(
    prompt: str = typer.Argument(..., help="The prompt/task for the agent to execute."),
    model: str = typer.Option("nvidia/nemotron-3-super-120b-a12b", help="The OpenRouter model to use.")
):
    """Run a single task using the agent loop."""
    console.print(f"[bold blue]Starting task:[/bold blue] {prompt}\n")
    
    # Construct initial history
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    system_instruction = (
        "You are an expert systems assistant with access to a bash terminal. "
        "Your task is to accomplish the user's goal by writing and running bash commands as needed. "
        "Verify your work whenever possible. "
        "Provide clear explanations before running actions, and report success once accomplished."
    )
    
    agent_loop(messages, model=model, system_instruction=system_instruction)

@app.command()
def interactive(
    model: str = typer.Option("nvidia/nemotron-3-super-120b-a12b", help="The OpenRouter model to use.")
):
    """Start an interactive chat session with the agent."""
    console.print(Panel(
        "Helen - the most beautiful agent\n"
        "Type your requests below. The agent will run bash commands to fulfill them.\n"
        "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to end the session.",
        title="[bold green]Interactive Mode[/bold green]",
        border_style="green"
    ))
    
    messages = []
    system_instruction = (
        "You are an expert systems assistant with access to a bash terminal. "
        "Your task is to accomplish the user's goal by writing and running bash commands as needed. "
        "This is an interactive chat, so you can ask follow-ups or maintain conversation. "
        "Verify your work whenever possible."
    )
    
    while True:
        try:
            prompt = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            if prompt.strip().lower() in ["exit", "quit"]:
                console.print("[bold yellow]Exiting. Goodbye![/bold yellow]")
                break
            
            # Append user prompt to history
            messages.append({"role": "user", "content": prompt})
            
            # Run agent loop (this will update history and output results)
            messages = agent_loop(messages, model=model, system_instruction=system_instruction)
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session interrupted. Goodbye![/bold yellow]")
            break

if __name__ == "__main__":
    app()
