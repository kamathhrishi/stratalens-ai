import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '@clerk/clerk-react'
import { useNavigate, Navigate } from 'react-router-dom'
import { Search, Building2 } from 'lucide-react'
import Sidebar from '../components/Sidebar'
import { searchCompanies, searchCompaniesPublic } from '../lib/api'

interface CompanyResult {
  symbol: string
  companyName: string
  sector?: string
  industry?: string
  marketCap?: number
  country?: string
}

function formatMarketCap(value?: number): string {
  if (!value) return ''
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`
  return `$${value.toLocaleString()}`
}

export default function CompaniesPage() {
  const { isSignedIn, getToken } = useAuth()
  const navigate = useNavigate()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<CompanyResult[]>([])
  const [suggestions, setSuggestions] = useState<CompanyResult[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const suggestionsRef = useRef<HTMLDivElement>(null)

  // Close suggestions on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Debounced autocomplete as user types
  const handleInputChange = useCallback(
    (value: string) => {
      setQuery(value)
      if (debounceRef.current) clearTimeout(debounceRef.current)

      if (value.trim().length < 1) {
        setSuggestions([])
        setShowSuggestions(false)
        return
      }

      debounceRef.current = setTimeout(async () => {
        try {
          const data = await searchCompaniesPublic(value.trim())
          setSuggestions(data.companies || [])
          setShowSuggestions(true)
        } catch {
          setSuggestions([])
        }
      }, 250)
    },
    []
  )

  const handleSearch = async () => {
    if (!query.trim() || isLoading) return
    setShowSuggestions(false)
    setIsLoading(true)
    setSearched(true)
    try {
      const token = await getToken()
      if (!token) return
      const data = await searchCompanies(query, token)
      setResults(data.companies || [])
    } catch (err) {
      console.error('Search failed:', err)
      setResults([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSuggestionClick = (company: CompanyResult) => {
    setShowSuggestions(false)
    navigate(`/companies/${company.symbol}`)
  }

  if (!isSignedIn) {
    return <Navigate to="/sign-in" replace />
  }

  return (
    <div className="min-h-screen bg-[#faf9f7]">
      <Sidebar
        isCollapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main
        className={`transition-all duration-200 ${
          sidebarCollapsed ? 'lg:ml-[72px]' : 'lg:ml-[240px]'
        }`}
      >
        <div className="max-w-[1200px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Header */}
          <div className="mb-6">
            <h1 className="text-2xl font-semibold text-[#0a1628]">Companies</h1>
            <p className="text-slate-500 mt-1">
              Search and explore company profiles
            </p>
          </div>

          {/* Search with autocomplete */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 mb-6">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 z-10" />
                <input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => handleInputChange(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSearch()
                    if (e.key === 'Escape') setShowSuggestions(false)
                  }}
                  onFocus={() => {
                    if (suggestions.length > 0) setShowSuggestions(true)
                  }}
                  placeholder="Search by ticker, company name, sector, or industry..."
                  className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-lg text-[#0a1628] placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0083f1]/20 focus:border-[#0083f1]"
                  disabled={isLoading}
                />

                {/* Suggestions dropdown */}
                {showSuggestions && suggestions.length > 0 && (
                  <div
                    ref={suggestionsRef}
                    className="absolute top-full left-0 right-0 mt-1 bg-white border border-slate-200 rounded-lg shadow-lg z-50 max-h-[360px] overflow-y-auto"
                  >
                    {suggestions.map((company) => (
                      <button
                        key={company.symbol}
                        onClick={() => handleSuggestionClick(company)}
                        className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-slate-50 transition-colors text-left border-b border-slate-100 last:border-0"
                      >
                        <div className="w-8 h-8 rounded-md bg-slate-100 flex items-center justify-center flex-shrink-0">
                          <Building2 className="w-4 h-4 text-slate-400" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-bold text-[#0083f1]">{company.symbol}</span>
                            <span className="text-sm text-[#0a1628] truncate">{company.companyName}</span>
                          </div>
                          <div className="text-xs text-slate-400 truncate">
                            {[company.sector, company.industry].filter(Boolean).join(' > ')}
                            {company.marketCap ? ` | ${formatMarketCap(company.marketCap)}` : ''}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button
                onClick={handleSearch}
                disabled={!query.trim() || isLoading}
                className="px-6 py-3 bg-[#0a1628] text-white rounded-lg hover:bg-[#1e293b] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                <Search className="w-5 h-5" />
                Search
              </button>
            </div>
          </div>

          {/* Results */}
          {isLoading && (
            <div className="text-center py-12 text-slate-500">Searching...</div>
          )}

          {!isLoading && results.length > 0 && (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {results.map((company) => (
                <button
                  key={company.symbol}
                  onClick={() => navigate(`/companies/${company.symbol}`)}
                  className="bg-white rounded-xl border border-slate-200 p-4 text-left hover:border-[#0083f1]/40 hover:shadow-sm transition-all group"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center flex-shrink-0 group-hover:bg-[#0083f1]/10 transition-colors">
                      <Building2 className="w-5 h-5 text-slate-500 group-hover:text-[#0083f1]" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-[#0083f1]">{company.symbol}</span>
                        <span className="text-xs text-slate-400">{formatMarketCap(company.marketCap)}</span>
                      </div>
                      <div className="text-sm font-medium text-[#0a1628] truncate">{company.companyName}</div>
                      {company.sector && (
                        <div className="text-xs text-slate-400 mt-1 truncate">
                          {company.sector}{company.industry ? ` > ${company.industry}` : ''}
                        </div>
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}

          {!isLoading && searched && results.length === 0 && (
            <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
              <p className="text-slate-500">No companies found matching "{query}"</p>
            </div>
          )}

          {!isLoading && !searched && (
            <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
              <Building2 className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-[#0a1628] mb-2">
                Search for a company
              </h3>
              <p className="text-slate-500 max-w-md mx-auto">
                Start typing a ticker or company name to see suggestions, or press
                Enter to search for more results.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
