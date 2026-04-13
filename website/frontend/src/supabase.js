import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || "https://hjgmfbwbwbqsrinuegzq.supabase.co";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || "sb_publishable_4rQ9j33LfyvTbcPjAF9vgg_I42Ka6XL";

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
