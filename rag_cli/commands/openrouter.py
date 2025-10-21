"""
OpenRouter CLI commands.

Provides commands for interacting with OpenRouter API:
- test: Test connection to OpenRouter
- models: List available models
- generate: Generate text from a prompt
- chat: Interactive chat with a model
"""

import json
import sys
import click

from Common.services.openrouter_service import openrouter_service


@click.group()
def openrouter():
    """OpenRouter API commands - unified access to multiple LLMs."""
    pass


@openrouter.command()
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def test(output_json):
    """Test connection to OpenRouter API."""
    if not openrouter_service.is_available():
        error_msg = (
            "❌ OpenRouter API key not configured.\n"
            "Set OPENROUTER_API_KEY in your .env file.\n"
            "Get your API key from: https://openrouter.ai/keys"
        )
        if output_json:
            click.echo(json.dumps({
                "status": "error",
                "message": "API key not configured"
            }))
        else:
            click.echo(error_msg, err=True)
        sys.exit(1)

    result = openrouter_service.test_connection()

    if output_json:
        click.echo(json.dumps(result, indent=2))
    else:
        if result["status"] == "success":
            click.echo(f"✅ {result['message']}")
            click.echo(f"⏱️  Response time: {result['response_time']}s")
            click.echo(f"🔐 Authenticated: {result['authenticated']}")
            click.echo(f"📊 API version: {result['api_version']}")
        else:
            click.echo(f"❌ Connection failed: {result['message']}", err=True)
            if result.get("error_type"):
                click.echo(f"   Error type: {result['error_type']}", err=True)


@openrouter.command()
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.option('--limit', type=int, help='Limit number of models shown')
@click.option('--search', type=str, help='Search models by name or ID')
def models(output_json, limit, search):
    """List available models from OpenRouter."""
    if not openrouter_service.is_available():
        error_msg = "❌ OpenRouter API key not configured."
        if output_json:
            click.echo(json.dumps({"status": "error", "message": error_msg}))
        else:
            click.echo(error_msg, err=True)
        sys.exit(1)

    click.echo("🔄 Fetching available models...", err=True)
    result = openrouter_service.get_available_models()

    if result["status"] != "success":
        if output_json:
            click.echo(json.dumps(result))
        else:
            click.echo(f"❌ Failed to get models: {result['message']}", err=True)
        sys.exit(1)

    models_list = result["models"]

    # Filter by search term
    if search:
        search_lower = search.lower()
        models_list = [
            m for m in models_list
            if search_lower in m["id"].lower() or search_lower in m.get("name", "").lower()
        ]

    # Apply limit
    if limit:
        models_list = models_list[:limit]

    if output_json:
        click.echo(json.dumps({
            "status": "success",
            "models": models_list,
            "count": len(models_list)
        }, indent=2))
    else:
        click.echo(f"\n✅ Found {len(models_list)} models:")
        click.echo("=" * 80)
        for model in models_list:
            click.echo(f"\n📦 {model['id']}")
            if model.get("name"):
                click.echo(f"   Name: {model['name']}")
            if model.get("description"):
                click.echo(f"   Description: {model['description']}")
            if model.get("context_length"):
                click.echo(f"   Context length: {model['context_length']:,} tokens")
            if model.get("pricing"):
                pricing = model["pricing"]
                if "prompt" in pricing:
                    click.echo(f"   Pricing: ${pricing.get('prompt')}/prompt token, ${pricing.get('completion')}/completion token")


@openrouter.command()
@click.option('--model', default='openai/gpt-3.5-turbo', help='Model to use (default: openai/gpt-3.5-turbo)')
@click.option('--max-tokens', type=int, default=250, help='Maximum tokens to generate')
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature (0.0-2.0)')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.argument('prompt')
def generate(model, max_tokens, temperature, output_json, prompt):
    """
    Generate text from a prompt.
    
    Example:
        rag_cli openrouter generate "Write a haiku about AI"
        rag_cli openrouter generate --model anthropic/claude-2 "Explain quantum computing"
    """
    if not openrouter_service.is_available():
        error_msg = "❌ OpenRouter API key not configured."
        if output_json:
            click.echo(json.dumps({"status": "error", "message": error_msg}))
        else:
            click.echo(error_msg, err=True)
        sys.exit(1)

    click.echo(f"🔄 Generating with {model}...", err=True)
    
    result = openrouter_service.generate_text(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )

    if output_json:
        click.echo(json.dumps(result, indent=2))
    else:
        if result["status"] == "success":
            click.echo("\n" + "=" * 80)
            click.echo(result["generated_text"])
            click.echo("=" * 80)
            click.echo(f"\n📊 Model: {result['model']}")
            if "usage" in result:
                usage = result["usage"]
                click.echo(f"📈 Tokens: {usage['total_tokens']} total ({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion)")
            if result.get("finish_reason"):
                click.echo(f"🏁 Finish reason: {result['finish_reason']}")
        else:
            click.echo(f"❌ Generation failed: {result['message']}", err=True)
            sys.exit(1)


@openrouter.command()
@click.option('--model', default='openai/gpt-3.5-turbo', help='Model to use')
@click.option('--max-tokens', type=int, default=250, help='Maximum tokens to generate')
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature')
@click.option('--system', type=str, help='System message to set context')
def chat(model, max_tokens, temperature, system):
    """
    Interactive chat with a model.
    
    Type your messages and press Enter. Type 'quit' or 'exit' to end the conversation.
    
    Example:
        rag_cli openrouter chat
        rag_cli openrouter chat --model anthropic/claude-2 --system "You are a helpful coding assistant"
    """
    if not openrouter_service.is_available():
        click.echo("❌ OpenRouter API key not configured.", err=True)
        sys.exit(1)

    click.echo(f"💬 Starting chat with {model}")
    click.echo("Type your message and press Enter. Type 'quit' or 'exit' to end.\n")

    messages = []
    
    # Add system message if provided
    if system:
        messages.append({"role": "system", "content": system})
        click.echo(f"📝 System: {system}\n")

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\n\n👋 Chat ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            click.echo("👋 Chat ended.")
            break

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Get response
        click.echo("🤔 Thinking...", err=True)
        result = openrouter_service.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if result["status"] == "success":
            response = result["response"]
            click.echo(f"\n{model}: {response}\n")
            
            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": response})
            
            # Show token usage
            if "usage" in result:
                usage = result["usage"]
                click.echo(f"📊 Tokens: {usage['total_tokens']} (↑{usage['prompt_tokens']} ↓{usage['completion_tokens']})", err=True)
        else:
            click.echo(f"❌ Error: {result['message']}", err=True)
            # Don't exit, allow user to try again


def register_openrouter_commands(cli):
    """Register OpenRouter commands with the CLI."""
    cli.add_command(openrouter)
