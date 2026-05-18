# Vector 4 Solutions — Final Year Project (CSIT321)

AI-powered aviation intelligence platform for passenger experience and safety on commercial flights. Uses computer vision and microphones to track and control lighting, temperature, and create alerts for on-flight crew.

## Project Structure

```
├── frontend/          React + Vite + Tailwind CSS
│   ├── src/
│   │   ├── App.jsx           Main app with all pages, navbar, routing
│   │   ├── Login.jsx         Authentication - sign in
│   │   ├── Signup.jsx        Authentication - create account
│   │   ├── Profile.jsx       User profile with metadata editing
│   │   ├── SubmitQuery.jsx   Support ticket submission
│   │   ├── BookAppointment.jsx  Consultation booking
│   │   ├── ProtectedRoute.jsx   Auth guard
│   │   ├── Logout.jsx        Sign out handler
│   │   ├── supabase.js       Supabase client (uses env vars)
│   │   ├── main.jsx          Entry point
│   │   ├── index.css         Global styles + fonts
│   │   └── App.css           Root styles
│   ├── .env                  Environment variables (not committed)
│   └── .env.example          Template for env vars
├── backend/           FastAPI backend
│   └── main.py        API endpoints
└── supabase_setup.sql SQL to create database tables
```

## Quick Start

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Database Setup
1. Go to your Supabase project dashboard
2. Open SQL Editor
3. Paste and run the contents of `supabase_setup.sql`
4. This creates `queries` and `appointments` tables with Row Level Security

## Environment Variables

Copy `.env.example` to `.env` in the frontend directory and fill in your Supabase credentials:

```
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Features

- **Persistent Navbar** with auth-aware navigation (login/signup vs profile/logout)
- **Scroll-reveal animations** on all sections
- **Full content pages** — About, Why Choose Us, Portfolio, Support with FAQs
- **Working forms** — Submit Query and Book Appointment write to Supabase (with graceful fallback)
- **Profile page** — displays and edits user metadata (enquiry type, profession)
- **Loading/error states** on all async operations
- **404 catch-all route**
- **Mobile-responsive** navigation and layouts
- **Supabase Auth** with email/password
- **Row Level Security** on all database tables
