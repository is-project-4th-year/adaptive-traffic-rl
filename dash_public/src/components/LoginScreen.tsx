import { useState } from 'react';
import { TrafficCone, Mail, Lock, Eye, EyeOff } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Checkbox } from './ui/checkbox';

interface LoginScreenProps {
  onLogin: () => void;
}

export function LoginScreen({ onLogin }: LoginScreenProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Simple validation
    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }
    
    // Mock authentication
    if (email && password) {
      onLogin();
    } else {
      setError('Invalid credentials. Please try again.');
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* Left Panel - 60% */}
      <div className="w-[60%] bg-gradient-to-br from-[#F8FAFC] via-white to-[#EFF6FF] flex items-center justify-center p-16">
        <div className="max-w-xl">
          {/* Logo & Title */}
          <div className="mb-8">
            <div className="inline-flex items-center justify-center size-16 rounded-2xl bg-gradient-to-br from-[#059669] to-[#047857] mb-6 shadow-lg">
              <TrafficCone className="size-8 text-white" />
            </div>
            <h1 className="text-[#0F172A] mb-3">
              Adaptive Traffic Signal Optimization
            </h1>
            <p className="text-[#475569] text-lg">
              RL-powered signal control for Nairobi
            </p>
          </div>

          {/* Illustration */}
          <div className="relative">
            <svg viewBox="0 0 400 300" className="w-full" aria-label="Traffic intersection illustration">
              {/* Intersection roads */}
              <rect x="0" y="130" width="400" height="40" fill="#E2E8F0" />
              <rect x="180" y="0" width="40" height="300" fill="#E2E8F0" />
              
              {/* Road markings */}
              <rect x="0" y="148" width="180" height="4" fill="white" opacity="0.8" />
              <rect x="220" y="148" width="180" height="4" fill="white" opacity="0.8" />
              <rect x="198" y="0" width="4" height="130" fill="white" opacity="0.8" />
              <rect x="198" y="170" width="4" height="130" fill="white" opacity="0.8" />
              
              {/* Traffic lights - Top (NS - Green) */}
              <g transform="translate(200, 90)">
                <rect x="-10" y="-40" width="20" height="50" rx="4" fill="#0F172A" />
                <circle cx="0" cy="-30" r="5" fill="#DC2626" opacity="0.3" />
                <circle cx="0" cy="-20" r="5" fill="#F59E0B" opacity="0.3" />
                <circle cx="0" cy="-10" r="5" fill="#059669" className="drop-shadow-lg" />
              </g>
              
              {/* Traffic lights - Right (EW - Baseline Gray) */}
              <g transform="translate(260, 150)">
                <rect x="-10" y="-10" width="50" height="20" rx="4" fill="#475569" />
                <circle cx="5" cy="0" r="5" fill="#DC2626" className="drop-shadow-md" />
                <circle cx="20" cy="0" r="5" fill="#F59E0B" opacity="0.3" />
                <circle cx="35" cy="0" r="5" fill="#64748B" opacity="0.3" />
              </g>
              
              {/* Cars */}
              <g transform="translate(100, 140)">
                <rect width="30" height="16" rx="3" fill="#059669" opacity="0.9" className="drop-shadow" />
                <rect x="8" y="3" width="6" height="10" rx="1" fill="#E2E8F0" opacity="0.8" />
                <rect x="16" y="3" width="6" height="10" rx="1" fill="#E2E8F0" opacity="0.8" />
              </g>
              
              <g transform="translate(280, 140)">
                <rect width="30" height="16" rx="3" fill="#64748B" opacity="0.7" />
                <rect x="8" y="3" width="6" height="10" rx="1" fill="#E2E8F0" opacity="0.5" />
                <rect x="16" y="3" width="6" height="10" rx="1" fill="#E2E8F0" opacity="0.5" />
              </g>
              
              {/* Comparison labels */}
              <g transform="translate(90, 230)">
                <text x="0" y="0" fill="#059669" fontSize="15" fontWeight="600">RL: Flowing ✓</text>
              </g>
              <g transform="translate(235, 230)">
                <text x="0" y="0" fill="#64748B" fontSize="15" fontWeight="600">Baseline: Queued</text>
              </g>
            </svg>
          </div>
        </div>
      </div>

      {/* Right Panel - 40% */}
      <div className="w-[40%] bg-white flex items-center justify-center p-16 shadow-2xl">
        <div className="w-full max-w-sm">
          <div className="mb-8">
            <h1 className="text-[#0F172A] mb-2">Sign In</h1>
            <p className="text-[#475569]">Access your dashboard</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email Input */}
            <div className="space-y-2">
              <Label htmlFor="email" className="text-[#0F172A]">
                Email Address
              </Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 size-5 text-[#64748B]" />
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                    setError('');
                  }}
                  className={`pl-10 h-11 bg-[#F8FAFC] border-[#E2E8F0] focus:border-[#059669] focus:ring-[#059669] ${error ? 'border-[#DC2626] focus:border-[#DC2626] focus:ring-[#DC2626]' : ''}`}
                  aria-invalid={!!error}
                  aria-describedby={error ? 'error-message' : undefined}
                />
              </div>
            </div>

            {/* Password Input */}
            <div className="space-y-2">
              <Label htmlFor="password" className="text-[#0F172A]">
                Password
              </Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 size-5 text-[#64748B]" />
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    setError('');
                  }}
                  className={`pl-10 pr-10 h-11 bg-[#F8FAFC] border-[#E2E8F0] focus:border-[#059669] focus:ring-[#059669] ${error ? 'border-[#DC2626] focus:border-[#DC2626] focus:ring-[#DC2626]' : ''}`}
                  aria-invalid={!!error}
                  aria-describedby={error ? 'error-message' : undefined}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[#64748B] hover:text-[#0F172A] transition-colors"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff className="size-5" /> : <Eye className="size-5" />}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div 
                id="error-message"
                className="p-3 bg-[#FEE2E2] border border-[#DC2626] rounded-lg flex items-start gap-2"
                role="alert"
              >
                <svg className="size-5 text-[#DC2626] flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <span className="text-[#DC2626]">{error}</span>
              </div>
            )}

            {/* Remember Me & Forgot Password */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="remember"
                  checked={rememberMe}
                  onCheckedChange={(checked) => setRememberMe(checked as boolean)}
                />
                <Label 
                  htmlFor="remember" 
                  className="text-[#475569] cursor-pointer"
                >
                  Remember me
                </Label>
              </div>
              <a 
                href="#" 
                className="text-[#3B82F6] hover:text-[#2563EB] transition-colors"
                onClick={(e) => e.preventDefault()}
              >
                Forgot password?
              </a>
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              className="w-full h-11 bg-gradient-to-r from-[#059669] to-[#047857] hover:from-[#047857] hover:to-[#065f46] text-white shadow-lg hover:shadow-xl transition-all duration-200"
            >
              Sign In
            </Button>

            {/* Footer */}
            <p className="text-center text-[#475569]">
              Need access?{' '}
              <a 
                href="#" 
                className="text-[#3B82F6] hover:text-[#2563EB] transition-colors font-medium"
                onClick={(e) => e.preventDefault()}
              >
                Contact admin
              </a>
            </p>
          </form>
        </div>
      </div>
    </div>
  );
}