import { useState, useEffect } from 'react'
import { useAuth } from '@clerk/clerk-react'
import { Navigate, useNavigate } from 'react-router-dom'
import {
  Briefcase,
  Eye,
  Trash2,
  Search,
  Building2,
  Plus,
  X
} from 'lucide-react'
import Sidebar from '../components/Sidebar'
import {
  fetchPortfolioSummary,
  addHolding,
  deleteHolding,
  addToWatchlist,
  removeFromWatchlist,
  searchCompaniesPublic,
  type PortfolioSummary,
  type Holding,
  type WatchlistItem,
} from '../lib/api'

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
}

function formatPrice(price: number | undefined): string {
  if (!price) return '—'
  return `$${price.toFixed(2)}`
}

export default function PortfolioPage() {
  const { isSignedIn, getToken } = useAuth()
  const navigate = useNavigate()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [activeTab, setActiveTab] = useState<'holdings' | 'watchlist'>('holdings')

  // Data states
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Add company modal states
  const [showAddModal, setShowAddModal] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [searchLoading, setSearchLoading] = useState(false)

  // Load portfolio data
  const loadPortfolio = async () => {
    setLoading(true)
    setError(null)
    try {
      const token = await getToken()
      if (!token) return
      const data = await fetchPortfolioSummary(token)
      setPortfolio(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load portfolio')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isSignedIn) {
      loadPortfolio()
    }
  }, [isSignedIn])

  const handleDelete = async (symbol: string, type: 'holdings' | 'watchlist') => {
    try {
      const token = await getToken()
      if (!token) return

      if (type === 'holdings') {
        await deleteHolding(symbol, token)
      } else {
        await removeFromWatchlist(symbol, token)
      }

      loadPortfolio()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete')
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearchLoading(true)
    try {
      console.log('🔍 Searching for:', searchQuery)
      const data = await searchCompaniesPublic(searchQuery)
      console.log('✅ Search results:', data)
      setSearchResults(data.companies || [])
    } catch (err) {
      console.error('❌ Search error:', err)
      alert(`Search failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setSearchLoading(false)
    }
  }

  const handleAddCompany = async (company: any) => {
    try {
      const token = await getToken()
      if (!token) return

      if (activeTab === 'holdings') {
        await addHolding(company.symbol, company.companyName, token)
      } else {
        await addToWatchlist(company.symbol, company.companyName, token)
      }

      setShowAddModal(false)
      setSearchQuery('')
      setSearchResults([])
      loadPortfolio()
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to add company'
      if (errorMsg.includes('409')) {
        alert(`${company.symbol} already exists in your ${activeTab}`)
      } else {
        alert(errorMsg)
      }
    }
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
        <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Header */}
          <div className="mb-6">
            <h1 className="text-2xl font-semibold text-[#0a1628] flex items-center gap-2">
              <Briefcase className="w-6 h-6" />
              Portfolio
            </h1>
            <p className="text-slate-500 mt-1">
              Manage your holdings and watchlist
            </p>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mb-6 border-b border-slate-200">
            <button
              onClick={() => setActiveTab('holdings')}
              className={`px-4 py-2 font-medium transition-colors relative ${
                activeTab === 'holdings'
                  ? 'text-[#0a1628]'
                  : 'text-slate-500 hover:text-[#0a1628]'
              }`}
            >
              <Briefcase className="w-4 h-4 inline mr-2" />
              Current Holdings
              {portfolio && (
                <span className="ml-2 text-xs bg-slate-100 px-2 py-0.5 rounded-full">
                  {portfolio.holdings_count}
                </span>
              )}
              {activeTab === 'holdings' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#0a1628]" />
              )}
            </button>
            <button
              onClick={() => setActiveTab('watchlist')}
              className={`px-4 py-2 font-medium transition-colors relative ${
                activeTab === 'watchlist'
                  ? 'text-[#0a1628]'
                  : 'text-slate-500 hover:text-[#0a1628]'
              }`}
            >
              <Eye className="w-4 h-4 inline mr-2" />
              Watchlist
              {portfolio && (
                <span className="ml-2 text-xs bg-slate-100 px-2 py-0.5 rounded-full">
                  {portfolio.watchlist_count}
                </span>
              )}
              {activeTab === 'watchlist' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#0a1628]" />
              )}
            </button>
          </div>

          {/* Add button */}
          <div className="mb-6">
            <button
              onClick={() => setShowAddModal(true)}
              className="px-4 py-2 bg-[#0a1628] text-white rounded-lg hover:bg-[#1e293b] transition-colors flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add to {activeTab === 'holdings' ? 'Holdings' : 'Watchlist'}
            </button>
          </div>

          {/* Content */}
          {loading && (
            <div className="text-center py-12 text-slate-500">Loading...</div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-600">
              {error}
            </div>
          )}

          {!loading && !error && portfolio && (
            <>
              {activeTab === 'holdings' && (
                <HoldingsList
                  holdings={portfolio.holdings}
                  onDelete={(symbol) => handleDelete(symbol, 'holdings')}
                  onNavigate={navigate}
                />
              )}
              {activeTab === 'watchlist' && (
                <WatchlistList
                  watchlist={portfolio.watchlist}
                  onDelete={(symbol) => handleDelete(symbol, 'watchlist')}
                  onNavigate={navigate}
                />
              )}
            </>
          )}
        </div>
      </main>

      {/* Add Company Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="p-6 border-b border-slate-200 flex items-center justify-between">
              <h2 className="text-xl font-semibold text-[#0a1628]">
                Add to {activeTab === 'holdings' ? 'Holdings' : 'Watchlist'}
              </h2>
              <button
                onClick={() => {
                  setShowAddModal(false)
                  setSearchQuery('')
                  setSearchResults([])
                }}
                className="p-1 hover:bg-slate-100 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 flex-1 overflow-y-auto">
              {/* Search */}
              <div className="flex gap-2 mb-4">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Search by ticker or company name..."
                  className="flex-1 px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0083f1]/20"
                />
                <button
                  onClick={handleSearch}
                  disabled={searchLoading}
                  className="px-4 py-2 bg-[#0a1628] text-white rounded-lg hover:bg-[#1e293b] transition-colors disabled:opacity-50"
                >
                  <Search className="w-5 h-5" />
                </button>
              </div>

              {/* Results */}
              {searchLoading && (
                <div className="text-center py-8 text-slate-500">Searching...</div>
              )}
              {!searchLoading && searchResults.length > 0 && (
                <div className="space-y-2">
                  {searchResults.map((company) => (
                    <button
                      key={company.symbol}
                      onClick={() => handleAddCompany(company)}
                      className="w-full flex items-center gap-3 p-3 border border-slate-200 rounded-lg hover:border-[#0083f1] hover:bg-slate-50 transition-all text-left"
                    >
                      <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center">
                        <Building2 className="w-5 h-5 text-slate-500" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-[#0083f1]">{company.symbol}</span>
                          <span className="text-sm text-[#0a1628]">{company.companyName}</span>
                        </div>
                        {company.sector && (
                          <div className="text-xs text-slate-400">
                            {company.sector}
                          </div>
                        )}
                      </div>
                      <Plus className="w-5 h-5 text-slate-400" />
                    </button>
                  ))}
                </div>
              )}
              {!searchLoading && searchQuery && searchResults.length === 0 && (
                <div className="text-center py-8 text-slate-500">
                  No results found
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function HoldingsList({
  holdings,
  onDelete,
  onNavigate
}: {
  holdings: Holding[]
  onDelete: (symbol: string) => void
  onNavigate: (path: string) => void
}) {
  if (holdings.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <Briefcase className="w-12 h-12 text-slate-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-[#0a1628] mb-2">
          No holdings yet
        </h3>
        <p className="text-slate-500">
          Add companies to track your portfolio
        </p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <th className="text-left px-6 py-3 text-xs font-medium text-slate-500 uppercase">Symbol</th>
              <th className="text-left px-6 py-3 text-xs font-medium text-slate-500 uppercase">Company</th>
              <th className="text-right px-6 py-3 text-xs font-medium text-slate-500 uppercase">Quantity</th>
              <th className="text-right px-6 py-3 text-xs font-medium text-slate-500 uppercase">Purchase Price</th>
              <th className="text-left px-6 py-3 text-xs font-medium text-slate-500 uppercase">Date Added</th>
              <th className="text-right px-6 py-3 text-xs font-medium text-slate-500 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {holdings.map((holding) => (
              <tr key={holding.id} className="hover:bg-slate-50 transition-colors">
                <td className="px-6 py-4">
                  <button
                    onClick={() => onNavigate(`/companies/${holding.symbol}`)}
                    className="font-bold text-[#0083f1] hover:underline"
                  >
                    {holding.symbol}
                  </button>
                </td>
                <td className="px-6 py-4 text-sm text-[#0a1628]">
                  {holding.company_name || '—'}
                </td>
                <td className="px-6 py-4 text-right text-sm text-[#0a1628]">
                  {holding.quantity?.toLocaleString() || '—'}
                </td>
                <td className="px-6 py-4 text-right text-sm text-[#0a1628]">
                  {formatPrice(holding.purchase_price)}
                </td>
                <td className="px-6 py-4 text-sm text-slate-500">
                  {formatDate(holding.created_at)}
                </td>
                <td className="px-6 py-4 text-right">
                  <button
                    onClick={() => onDelete(holding.symbol)}
                    className="p-1.5 text-red-500 hover:bg-red-50 rounded transition-colors"
                    title="Remove from holdings"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function WatchlistList({
  watchlist,
  onDelete,
  onNavigate
}: {
  watchlist: WatchlistItem[]
  onDelete: (symbol: string) => void
  onNavigate: (path: string) => void
}) {
  if (watchlist.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <Eye className="w-12 h-12 text-slate-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-[#0a1628] mb-2">
          No companies in watchlist
        </h3>
        <p className="text-slate-500">
          Add companies you want to monitor
        </p>
      </div>
    )
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {watchlist.map((item) => (
        <div
          key={item.id}
          className="bg-white rounded-xl border border-slate-200 p-4 hover:shadow-sm transition-all"
        >
          <div className="flex items-start justify-between mb-3">
            <button
              onClick={() => onNavigate(`/companies/${item.symbol}`)}
              className="flex items-center gap-2 group"
            >
              <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center group-hover:bg-[#0083f1]/10 transition-colors">
                <Building2 className="w-5 h-5 text-slate-500 group-hover:text-[#0083f1]" />
              </div>
              <div>
                <div className="font-bold text-[#0083f1] group-hover:underline">
                  {item.symbol}
                </div>
                <div className="text-xs text-slate-500">
                  {formatDate(item.created_at)}
                </div>
              </div>
            </button>
            <button
              onClick={() => onDelete(item.symbol)}
              className="p-1.5 text-red-500 hover:bg-red-50 rounded transition-colors"
              title="Remove from watchlist"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
          <div className="text-sm text-[#0a1628] truncate">
            {item.company_name || '—'}
          </div>
          {item.notes && (
            <div className="text-xs text-slate-400 mt-2 line-clamp-2">
              {item.notes}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
