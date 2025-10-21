"""
HuggingFace CLI commands for AI model interactions.

Supports both local models (sentence-transformers, transformers) 
and HuggingFace API integration.
"""

import click
import sys
from typing import Optional
from Common.services.huggingface_service import huggingface_service


@click.group()
def huggingface():
    """HuggingFace AI model commands (local & API)."""
    pass


@huggingface.command()
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def test(output_json: bool):
    """Test connection to local models and HuggingFace API."""
    if not huggingface_service.is_available():
        click.secho(
            "❌ HuggingFace service not available\n"
            "Install dependencies:\n"
            "  - For local embeddings: pip install sentence-transformers\n"
            "  - For local generation: pip install transformers torch\n"
            "  - Or set HUGGINGFACE_TOKEN in .env for API access",
            fg='red',
            err=True
        )
        raise click.ClickException("HuggingFace service unavailable")
    
    result = huggingface_service.test_connection()

    if output_json:
        import json
        click.echo(json.dumps(result, indent=2))
    else:
        click.secho(f"\n{'='*60}", fg='blue')
        click.secho("🤖 HuggingFace Service Status", fg='cyan', bold=True)
        click.secho(f"{'='*60}\n", fg='blue')
        
        # Overall status
        if result["status"] == "success":
            click.secho(f"✅ Status: {result['message']}", fg='green')
        elif result["status"] == "partial":
            click.secho(f"⚠️  Status: {result['message']}", fg='yellow')
        else:
            click.secho(f"❌ Status: {result['message']}", fg='red')
        
        # Local models
        click.secho(f"\n📦 Local Models:", fg='cyan', bold=True)
        local_models = result.get("local_models", {})
        
        for lib_name, info in local_models.items():
            status = info.get("status")
            if status == "available":
                click.secho(f"  ✅ {lib_name}: Available", fg='green')
                if "model" in info:
                    click.echo(f"     Model: {info['model']}")
                if "embedding_dim" in info:
                    click.echo(f"     Dimensions: {info['embedding_dim']}")
            elif status == "not_installed":
                click.secho(f"  ❌ {lib_name}: Not installed", fg='red')
                click.echo(f"     {info.get('message', '')}")
            else:
                click.secho(f"  ⚠️  {lib_name}: Error", fg='yellow')
                click.echo(f"     {info.get('error', 'Unknown error')}")
        
        # API connection
        click.secho(f"\n🌐 API Connection:", fg='cyan', bold=True)
        api_info = result.get("api_connection", {})
        api_status = api_info.get("status")
        
        if api_status == "connected":
            click.secho(f"  ✅ Connected", fg='green')
            click.echo(f"     Response time: {api_info.get('response_time', 'N/A')}s")
            auth_status = "Yes" if api_info.get('authenticated') else "No (public access)"
            click.echo(f"     Authenticated: {auth_status}")
        else:
            click.secho(f"  ❌ Not connected", fg='red')
            click.echo(f"     Error: {api_info.get('error', 'Unknown')}")
            click.echo(f"     💡 Set HUGGINGFACE_TOKEN in .env for API access")
        
        click.echo()


@huggingface.command()
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def capabilities(output_json: bool):
    """Show available capabilities and configured models."""
    caps = huggingface_service.get_capabilities()
    
    if output_json:
        import json
        click.echo(json.dumps(caps, indent=2))
    else:
        click.secho("\n🔧 HuggingFace Service Capabilities:", fg='cyan', bold=True)
        click.echo()
        
        status_icon = lambda x: "✅" if x else "❌"
        
        click.echo(f"{status_icon(caps['local_embeddings'])} Local embeddings: "
                   f"{'Available' if caps['local_embeddings'] else 'Not available'}")
        click.echo(f"{status_icon(caps['local_generation'])} Local generation: "
                   f"{'Available' if caps['local_generation'] else 'Not available'}")
        click.echo(f"{status_icon(caps['api_access'])} API access: "
                   f"{'Available' if caps['api_access'] else 'Not available'}")
        
        if caps.get('configured_embedding_model'):
            click.echo(f"\n📝 Configured embedding model: {caps['configured_embedding_model']}")
        click.echo()


@huggingface.command()
@click.argument('prompt')
@click.option('--model', default='microsoft/DialoGPT-medium',
              help='Model to use for text generation')
@click.option('--max-new-tokens', type=int, default=250,
              help='Maximum number of new tokens to generate')
@click.option('--temperature', type=float, default=0.7,
              help='Temperature for text generation (0.0-1.0)')
@click.option('--top-p', type=float, default=0.95,
              help='Top-p sampling parameter (0.0-1.0)')
@click.option('--local', 'use_local', is_flag=True,
              help='Force local model usage (requires transformers)')
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def generate(prompt: str, model: str, max_new_tokens: int, temperature: float, 
             top_p: float, use_local: bool, output_json: bool):
    """Generate text using a HuggingFace model (local or API)."""
    if not huggingface_service.is_available():
        click.secho("❌ HuggingFace service not available", fg='red', err=True)
        raise click.ClickException("Service unavailable")
    
    result = huggingface_service.generate_text(
        prompt=prompt,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_local=use_local
    )

    if output_json:
        import json
        click.echo(json.dumps(result, indent=2))
    else:
        if result["status"] == "success":
            click.secho("\n📝 Generated Text:", fg='blue', bold=True)
            click.echo("-" * 60)
            click.echo(result["generated_text"])
            click.echo("-" * 60)
            click.echo(f"\n🤖 Model: {result['model']}")
            click.echo(f"🔧 Backend: {result.get('backend', 'unknown')}")
            if result.get('device'):
                click.echo(f"💻 Device: {result['device']}")
            click.echo()
        else:
            click.secho(f"\n❌ Text generation failed:", fg='red', err=True)
            click.echo(f"{result['message']}", err=True)
            
            if use_local:
                click.echo("\n💡 Tips:", err=True)
                click.echo("  - Install transformers: pip install transformers torch", err=True)
                click.echo("  - Try without --local flag to use API", err=True)
            click.echo()
            raise click.ClickException(result['message'])


@huggingface.command()
@click.argument('texts', nargs=-1, required=False)
@click.option('--model', default=None,
              help='Model to use for embeddings (default: configured model)')
@click.option('--api', 'use_api', is_flag=True,
              help='Force API usage instead of local model')
@click.option('--json', 'output_json', is_flag=True,
              help='Output in JSON format')
@click.option('--stdin', is_flag=True,
              help='Read text from stdin instead of arguments')
def embeddings(texts, model, use_api, output_json, stdin):
    """Get embeddings for text using local or API models."""
    if not huggingface_service.is_available():
        click.secho("❌ HuggingFace service not available", fg='red', err=True)
        raise click.ClickException("Service unavailable")
    
    if stdin:
        # Read from stdin
        input_text = sys.stdin.read().strip()
        if not input_text:
            click.echo("Error: No text provided via stdin", err=True)
            return
        texts = [input_text]
    elif not texts:
        click.echo("Error: No texts provided. Use --stdin to read from stdin or provide text as arguments", err=True)
        return

    # Get embeddings (prefer local unless --api specified)
    def progress_callback(message: str):
        if not output_json:
            click.echo(f"🔄 {message}")
    
    if use_api:
        result = huggingface_service.get_embeddings(
            list(texts), 
            model=model, 
            use_local=False
        )
    else:
        # Use local with progress feedback
        result = huggingface_service.get_local_embeddings(
            list(texts), 
            model=model,
            progress_callback=progress_callback
        )

    if output_json:
        import json
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if result["status"] == "success":
            click.secho(f"\n✅ Generated embeddings for {result['text_count']} text(s)", fg='green')
            click.echo(f"🤖 Model: {result['model']}")
            click.echo(f"🔧 Backend: {result.get('backend', 'unknown')}")
            click.echo(f" Embedding dimensions: {result['dimensions']}")
            
            # Show batch processing info if available
            if result.get('batches_processed'):
                click.echo(f"📦 Batches processed: {result['batches_processed']}")
            
            # Show validation warnings if any
            if result.get('validation_warnings'):
                click.secho(f"⚠️  Validation warnings: {result['validation_warnings']}", fg='yellow')
            
            # Show first few values of first embedding
            if result['embeddings']:
                first_embedding = result['embeddings'][0]
                preview = first_embedding[:5] if len(first_embedding) > 5 else first_embedding
                preview_str = ", ".join(f"{v:.4f}" for v in preview)
                click.echo(f"🔢 First embedding preview: [{preview_str}...]")
            click.echo()
        else:
            click.secho(f"\n❌ Embedding generation failed:", fg='red', err=True)
            click.echo(f"{result['message']}", err=True)
            
            click.echo("\n💡 Tips:", err=True)
            click.echo("  - For local: pip install sentence-transformers", err=True)
            click.echo("  - For API: set HUGGINGFACE_TOKEN and use --api flag", err=True)
            click.echo()
            raise click.ClickException(result['message'])


@huggingface.command()
@click.option('--search', default='', help='Search term for models')
@click.option('--limit', type=int, default=10, help='Maximum number of models to list')
@click.option('--filter', 'filter_by', default=None, 
              help='Filter by task (e.g., text-generation, feature-extraction)')
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def models(search: str, limit: int, filter_by: Optional[str], output_json: bool):
    """List available models on HuggingFace Hub."""
    result = huggingface_service.list_models(
        search=search, 
        limit=limit,
        filter_by=filter_by
    )

    if output_json:
        import json
        click.echo(json.dumps(result, indent=2))
    else:
        if result["status"] == "success":
            filters_str = f" (filter: {filter_by})" if filter_by else ""
            search_str = f" matching '{search}'" if search else ""
            
            click.secho(f"\n🤖 Available Models{search_str}{filters_str}:", fg='blue', bold=True)
            click.echo(f"Found {result['count']} models\n")
            
            for idx, model in enumerate(result["models"][:limit], 1):
                click.secho(f"{idx}. {model.get('id', 'Unknown')}", fg='cyan', bold=True)
                if model.get('downloads'):
                    click.echo(f"   📥 Downloads: {model['downloads']:,}")
                if model.get('likes'):
                    click.echo(f"   ❤️  Likes: {model['likes']:,}")
                if model.get('pipeline_tag'):
                    click.echo(f"   🏷️  Task: {model['pipeline_tag']}")
                click.echo()
        else:
            click.secho(f"❌ Failed to list models: {result['message']}", fg='red', err=True)
            click.echo("\n💡 Tip: Set HUGGINGFACE_TOKEN in .env for authenticated access", err=True)
            raise click.ClickException(result['message'])