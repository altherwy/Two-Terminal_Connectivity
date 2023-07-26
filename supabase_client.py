
#%%
import os
from supabase import create_client, Client # type: ignore

url: str = os.environ.get("SUPABASE_URL") # type: ignore
key: str = os.environ.get("SUPABASE_KEY") # type: ignore
supabase: Client = create_client(url, key)
# %%
