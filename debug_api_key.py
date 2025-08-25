#!/usr/bin/env python3
"""
Debug script to check Databento API key configuration
"""
import os
from dotenv import load_dotenv

def check_api_key():
    print("🔍 Checking Databento API Key Configuration...")
    
    # Load .env file if it exists
    load_dotenv()
    
    # Check environment variable
    api_key = os.getenv("DATABENTO_API_KEY")
    
    if not api_key:
        print("❌ DATABENTO_API_KEY not found in environment variables")
        print("\n📋 To fix this:")
        print("1. Create a .env file in your project root with:")
        print("   DATABENTO_API_KEY=your_api_key_here")
        print("2. Or export the environment variable:")
        print("   export DATABENTO_API_KEY=your_api_key_here")
        print("3. Or set it in PowerShell:")
        print("   $env:DATABENTO_API_KEY='your_api_key_here'")
        return False
    
    # Check if key looks valid (starts with expected prefix)
    if len(api_key) < 10:
        print(f"⚠️  API key looks too short: '{api_key[:5]}...' (length: {len(api_key)})")
        return False
    
    # Mask the key for security
    masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:]
    print(f"✅ DATABENTO_API_KEY found: {masked_key}")
    print(f"📏 Key length: {len(api_key)} characters")
    
    # Test a simple API call
    print("\n🔬 Testing API connection...")
    try:
        import databento as db
        client = db.Historical(api_key)
        print("✅ Databento client initialized successfully")
        
        # Try a minimal API call (metadata request)
        try:
            # This is a very lightweight call to test auth
            datasets = client.metadata.list_datasets()
            print(f"✅ API authentication successful - found {len(datasets)} datasets")
            return True
        except Exception as e:
            print(f"❌ API call failed: {e}")
            if "401" in str(e) or "auth" in str(e).lower():
                print("🔑 This appears to be an authentication issue")
                print("   - Check that your API key is correct")
                print("   - Verify your Databento account is active")
                print("   - Visit: https://databento.com/docs/portal/api-keys")
            return False
            
    except ImportError:
        print("❌ databento package not installed")
        print("💡 Install with: pip install databento")
        return False
    except Exception as e:
        print(f"❌ Error creating client: {e}")
        return False

if __name__ == "__main__":
    success = check_api_key()
    if success:
        print("\n🎉 API key configuration is working correctly!")
    else:
        print("\n🚨 Please fix the API key configuration before proceeding")
    
    # Check current working directory for .env
    print(f"\n📁 Current directory: {os.getcwd()}")
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        print(f"✅ .env file found at: {env_path}")
    else:
        print(f"❌ No .env file found at: {env_path}")
