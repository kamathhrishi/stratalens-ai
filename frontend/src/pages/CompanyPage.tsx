import { useState, useEffect, useCallback } from 'react'
import { useParams, Navigate } from 'react-router-dom'
import { useAuth } from '@clerk/clerk-react'
import {
  Building2,
  TrendingUp,
  TrendingDown,
  Globe,
  Users,
  Calendar,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  FileText,
  Briefcase,
  Eye,
} from 'lucide-react'
import Sidebar from '../components/Sidebar'
import DocumentPanel, { DOCUMENT_PANEL_WIDTH } from '../components/DocumentPanel'
import type { DocumentPanelContent } from '../components/DocumentPanel'
import {
  fetchCompanyProfile,
  fetchIncomeStatement,
  fetchBalanceSheet,
  fetchCashFlow,
  fetchProductSegments,
  fetchGeographicSegments,
  fetchAvailableTranscripts,
  fetchTranscript,
  fetchAvailableSECFilings,
  addHolding,
  addToWatchlist,
} from '../lib/api'

// ── Helpers ──

function fmt(value: unknown, style: 'currency' | 'number' | 'percent' = 'number'): string {
  if (value === null || value === undefined) return '—'
  const n = typeof value === 'number' ? value : Number(value)
  if (isNaN(n)) return String(value)

  if (style === 'percent') return `${n.toFixed(2)}%`

  const abs = Math.abs(n)
  const sign = n < 0 ? '-' : ''
  const prefix = style === 'currency' ? '$' : ''

  if (abs >= 1e12) return `${sign}${prefix}${(abs / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${sign}${prefix}${(abs / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${sign}${prefix}${(abs / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${sign}${prefix}${(abs / 1e3).toFixed(2)}K`
  if (Number.isInteger(n)) return `${sign}${prefix}${abs.toLocaleString()}`
  return `${sign}${prefix}${abs.toFixed(2)}`
}

// ── Types ──

interface CompanyData {
  [key: string]: unknown
}

interface FinancialStatement {
  [key: string]: unknown
}

interface Segment {
  segment: string
  revenue: number
  formatted_revenue: string
  percentage: number
  formatted_percentage: string
}

interface TranscriptEntry {
  year: number
  quarter: number
  date: string | null
  company_name: string
}

interface SECFilingEntry {
  ticker: string
  filing_type: string
  fiscal_year: number
  quarter?: number | null
  filing_date?: string | null
  filing_period?: string | null
  document_length?: number
}

// ── Main Component ──

export default function CompanyPage() {
  const { symbol } = useParams<{ symbol: string }>()
  const { isSignedIn, getToken } = useAuth()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Data states
  const [company, setCompany] = useState<CompanyData | null>(null)
  const [incomeStatements, setIncomeStatements] = useState<FinancialStatement[]>([])
  const [balanceSheets, setBalanceSheets] = useState<FinancialStatement[]>([])
  const [cashFlows, setCashFlows] = useState<FinancialStatement[]>([])
  const [productSegments, setProductSegments] = useState<Segment[]>([])
  const [geoSegments, setGeoSegments] = useState<Segment[]>([])
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([])
  const [secFilings, setSECFilings] = useState<SECFilingEntry[]>([])

  // UI states
  const [loading, setLoading] = useState(true)
  const [detailsLoading, setDetailsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [financialTab, setFinancialTab] = useState<'income' | 'balance' | 'cashflow'>('income')
  const [expandedTranscript, setExpandedTranscript] = useState<string | null>(null)
  const [transcriptTexts, setTranscriptTexts] = useState<Record<string, string>>({})
  const [transcriptLoading, setTranscriptLoading] = useState<string | null>(null)
  const [aboutExpanded, setAboutExpanded] = useState(false)
  const [addingToPortfolio, setAddingToPortfolio] = useState(false)
  const [documentPanel, setDocumentPanel] = useState<{ open: boolean; content: DocumentPanelContent | null }>({
    open: false,
    content: null,
  })

  // Phase 1: Load profile first (shows header + about immediately)
  const loadProfile = useCallback(async () => {
    if (!symbol) return
    setLoading(true)
    setError(null)
    try {
      const token = await getToken()
      if (!token) return

      const profileData = await fetchCompanyProfile(symbol, token)
      if (profileData.success) {
        setCompany(profileData.company)
      } else {
        setError('Company not found')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load company data')
    } finally {
      setLoading(false)
    }
  }, [symbol, getToken])

  // Phase 2: Load financials, segments, transcripts in parallel (non-blocking)
  const loadDetails = useCallback(async () => {
    if (!symbol) return
    setDetailsLoading(true)
    try {
      const token = await getToken()
      if (!token) return

      const [incomeRes, balanceRes, cashRes, prodSegRes, geoSegRes, transcriptRes, secFilingsRes] =
        await Promise.allSettled([
          fetchIncomeStatement(symbol, token),
          fetchBalanceSheet(symbol, token),
          fetchCashFlow(symbol, token),
          fetchProductSegments(symbol, token),
          fetchGeographicSegments(symbol, token),
          fetchAvailableTranscripts(symbol, token),
          fetchAvailableSECFilings(symbol, token),
        ])

      if (incomeRes.status === 'fulfilled') setIncomeStatements(incomeRes.value.statements || [])
      if (balanceRes.status === 'fulfilled') setBalanceSheets(balanceRes.value.statements || [])
      if (cashRes.status === 'fulfilled') setCashFlows(cashRes.value.statements || [])
      if (prodSegRes.status === 'fulfilled') setProductSegments(prodSegRes.value.segments || [])
      if (geoSegRes.status === 'fulfilled') setGeoSegments(geoSegRes.value.segments || [])
      if (transcriptRes.status === 'fulfilled') setTranscripts(transcriptRes.value.transcripts || [])
      if (secFilingsRes.status === 'fulfilled') setSECFilings(secFilingsRes.value.filings || [])
    } catch {
      // Non-critical — profile is already shown
    } finally {
      setDetailsLoading(false)
    }
  }, [symbol, getToken])

  useEffect(() => {
    if (isSignedIn) {
      loadProfile()
      loadDetails()
    }
  }, [isSignedIn, loadProfile, loadDetails])

  const handleAddToPortfolio = async (type: 'holdings' | 'watchlist') => {
    if (!symbol || !company) return
    setAddingToPortfolio(true)
    try {
      const token = await getToken()
      if (!token) return

      const companyName = String(company.companyName || '')

      if (type === 'holdings') {
        await addHolding(symbol, companyName, token)
        alert(`${symbol} added to your holdings!`)
      } else {
        await addToWatchlist(symbol, companyName, token)
        alert(`${symbol} added to your watchlist!`)
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to add'
      if (errorMsg.includes('409')) {
        alert(`${symbol} already exists in your ${type}`)
      } else {
        alert(errorMsg)
      }
    } finally {
      setAddingToPortfolio(false)
    }
  }

  const handleTranscriptToggle = async (year: number, quarter: number) => {
    const key = `${year}-Q${quarter}`
    if (expandedTranscript === key) {
      setExpandedTranscript(null)
      return
    }
    setExpandedTranscript(key)

    if (transcriptTexts[key]) return

    setTranscriptLoading(key)
    try {
      const token = await getToken()
      if (!token) return
      const data = await fetchTranscript(symbol!, year, quarter, token)
      setTranscriptTexts((prev) => ({ ...prev, [key]: data.transcript_text || data.transcript || '' }))
    } catch {
      setTranscriptTexts((prev) => ({ ...prev, [key]: 'Failed to load transcript.' }))
    } finally {
      setTranscriptLoading(null)
    }
  }

  if (!isSignedIn) {
    return <Navigate to="/sign-in" replace />
  }

  // ── Derive display values from company data ──

  const price = company?.price as number | undefined
  const change = company?.changes as number | undefined
  const changePct = price && change ? ((change / (price - change)) * 100) : undefined
  const isPositive = (change ?? 0) >= 0

  return (
    <div className="min-h-screen bg-[#faf9f7]">
      <Sidebar
        isCollapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main
        style={{
          marginLeft: sidebarCollapsed ? '60px' : '220px',
          marginRight: documentPanel.open ? `${DOCUMENT_PANEL_WIDTH}px` : '0px',
          transition: 'margin 0.3s',
        }}
      >
        <div className="max-w-[1200px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
          {loading && (
            <div className="text-center py-20 text-slate-500">Loading company data...</div>
          )}

          {error && !loading && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
              <p className="text-red-600">{error}</p>
            </div>
          )}

          {!loading && company && (
            <>
              {/* ── 1. Header ── */}
              <div className="bg-white rounded-xl border border-slate-200 p-5">
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col sm:flex-row sm:items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-[#0a1628] flex items-center justify-center flex-shrink-0">
                      <Building2 className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="px-2 py-0.5 bg-[#0083f1]/10 text-[#0083f1] text-sm font-bold rounded">
                          {symbol}
                        </span>
                        <span className="text-xl font-semibold text-[#0a1628] truncate">
                          {String(company.companyName || '')}
                        </span>
                      </div>
                      <div className="text-sm text-slate-400 mt-0.5">
                        {String(company.sector || '')}{company.industry ? ` > ${String(company.industry)}` : ''}
                        {company.exchangeShortName ? ` | ${String(company.exchangeShortName)}` : ''}
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0">
                      {price != null && (
                        <div className="text-2xl font-semibold text-[#0a1628]">${price.toFixed(2)}</div>
                      )}
                      {change != null && (
                        <div className={`flex items-center justify-end gap-1 text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-500'}`}>
                          {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                          {isPositive ? '+' : ''}{change.toFixed(2)}
                          {changePct != null && ` (${isPositive ? '+' : ''}${changePct.toFixed(2)}%)`}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Portfolio Actions */}
                  <div className="flex gap-2 pt-2 border-t border-slate-100">
                    <button
                      onClick={() => handleAddToPortfolio('holdings')}
                      disabled={addingToPortfolio}
                      className="flex items-center gap-2 px-4 py-2 bg-[#0a1628] text-white rounded-lg hover:bg-[#1e293b] transition-colors disabled:opacity-50 text-sm"
                    >
                      <Briefcase className="w-4 h-4" />
                      Add to Holdings
                    </button>
                    <button
                      onClick={() => handleAddToPortfolio('watchlist')}
                      disabled={addingToPortfolio}
                      className="flex items-center gap-2 px-4 py-2 border border-slate-200 text-[#0a1628] rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50 text-sm"
                    >
                      <Eye className="w-4 h-4" />
                      Add to Watchlist
                    </button>
                  </div>
                </div>
              </div>

              {/* ── 2. Key Metrics Row ── */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                {[
                  { label: 'Market Cap', value: fmt(company.mktCap, 'currency') },
                  { label: 'P/E', value: fmt(company.priceEarningsRatio ?? company.peRatio) },
                  { label: 'EPS', value: fmt(company.epsDiluted, 'currency') },
                  { label: 'Revenue', value: fmt(company.revenue, 'currency') },
                  { label: 'Net Income', value: fmt(company.netIncome, 'currency') },
                  { label: 'Gross Margin', value: company.grossProfitMargin != null ? fmt((company.grossProfitMargin as number) * 100, 'percent') : '—' },
                ].map((m) => (
                  <div key={m.label} className="bg-white rounded-xl border border-slate-200 p-3">
                    <div className="text-xs text-slate-400 mb-1">{m.label}</div>
                    <div className="text-sm font-semibold text-[#0a1628]">{m.value}</div>
                  </div>
                ))}
              </div>

              {/* ── 3. About ── */}
              <div className="bg-white rounded-xl border border-slate-200 p-5">
                <h2 className="text-lg font-semibold text-[#0a1628] mb-3">About</h2>
                {company.description ? (
                  <div className="mb-3">
                    <p className={`text-sm text-slate-600 leading-relaxed ${!aboutExpanded ? 'line-clamp-3' : ''}`}>
                      {String(company.description)}
                    </p>
                    <button
                      onClick={() => setAboutExpanded(!aboutExpanded)}
                      className="text-sm text-[#0083f1] hover:underline mt-1"
                    >
                      {aboutExpanded ? 'Show less' : 'Read more'}
                    </button>
                  </div>
                ) : null}
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-y-3 gap-x-6 text-sm">
                  {company.ceo ? (
                    <div>
                      <span className="text-slate-400">CEO</span>
                      <div className="font-medium text-[#0a1628]">{String(company.ceo)}</div>
                    </div>
                  ) : null}
                  {company.fullTimeEmployees ? (
                    <div>
                      <span className="text-slate-400">Employees</span>
                      <div className="font-medium text-[#0a1628] flex items-center gap-1">
                        <Users className="w-3.5 h-3.5" />
                        {Number(company.fullTimeEmployees).toLocaleString()}
                      </div>
                    </div>
                  ) : null}
                  {(company.city || company.state || company.country) ? (
                    <div>
                      <span className="text-slate-400">Headquarters</span>
                      <div className="font-medium text-[#0a1628] flex items-center gap-1">
                        <Globe className="w-3.5 h-3.5" />
                        {[company.city, company.state, company.country].filter(Boolean).map(String).join(', ')}
                      </div>
                    </div>
                  ) : null}
                  {company.ipoDate ? (
                    <div>
                      <span className="text-slate-400">IPO Date</span>
                      <div className="font-medium text-[#0a1628] flex items-center gap-1">
                        <Calendar className="w-3.5 h-3.5" />
                        {String(company.ipoDate)}
                      </div>
                    </div>
                  ) : null}
                  {company.website ? (
                    <div>
                      <span className="text-slate-400">Website</span>
                      <div className="font-medium text-[#0083f1]">
                        <a
                          href={String(company.website)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:underline flex items-center gap-1"
                        >
                          {String(company.website).replace(/^https?:\/\/(www\.)?/, '')}
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>

              {/* ── 4. Financial Statements ── */}
              {detailsLoading && (
                <div className="bg-white rounded-xl border border-slate-200 p-8 text-center">
                  <div className="animate-pulse flex flex-col items-center gap-2">
                    <div className="h-4 w-48 bg-slate-200 rounded" />
                    <div className="h-3 w-32 bg-slate-100 rounded" />
                  </div>
                  <p className="text-sm text-slate-400 mt-3">Loading financials...</p>
                </div>
              )}
              {!detailsLoading && (incomeStatements.length > 0 || balanceSheets.length > 0 || cashFlows.length > 0) && (
                <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                  <div className="flex border-b border-slate-200">
                    {(['income', 'balance', 'cashflow'] as const).map((tab) => (
                      <button
                        key={tab}
                        onClick={() => setFinancialTab(tab)}
                        className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                          financialTab === tab
                            ? 'text-[#0083f1] border-b-2 border-[#0083f1] bg-[#0083f1]/5'
                            : 'text-slate-500 hover:text-[#0a1628] hover:bg-slate-50'
                        }`}
                      >
                        {tab === 'income' ? 'Income Statement' : tab === 'balance' ? 'Balance Sheet' : 'Cash Flow'}
                      </button>
                    ))}
                  </div>

                  <div className="overflow-x-auto">
                    <FinancialTable
                      tab={financialTab}
                      income={incomeStatements}
                      balance={balanceSheets}
                      cashflow={cashFlows}
                    />
                  </div>
                </div>
              )}

              {/* ── 5. Revenue Segments ── */}
              {(productSegments.length > 0 || geoSegments.length > 0) && (
                <div className="grid md:grid-cols-2 gap-4">
                  {productSegments.length > 0 && (
                    <SegmentCard title="Product Segments" segments={productSegments} />
                  )}
                  {geoSegments.length > 0 && (
                    <SegmentCard title="Geographic Segments" segments={geoSegments} />
                  )}
                </div>
              )}

              {/* ── 6. Earnings Transcripts ── */}
              {transcripts.length > 0 && (
                <div className="bg-white rounded-xl border border-slate-200 p-5">
                  <h2 className="text-lg font-semibold text-[#0a1628] mb-3">Earnings Transcripts</h2>
                  <div className="space-y-2">
                    {transcripts.map((t) => {
                      const key = `${t.year}-Q${t.quarter}`
                      const isOpen = expandedTranscript === key
                      const isLoadingThis = transcriptLoading === key
                      return (
                        <div key={key} className="border border-slate-200 rounded-lg overflow-hidden">
                          <button
                            onClick={() => handleTranscriptToggle(t.year, t.quarter)}
                            className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-50 transition-colors"
                          >
                            <div className="flex items-center gap-3">
                              <FileText className="w-4 h-4 text-slate-400" />
                              <span className="text-sm font-medium text-[#0a1628]">
                                Q{t.quarter} {t.year}
                              </span>
                              {t.date && (
                                <span className="text-xs text-slate-400">{t.date}</span>
                              )}
                            </div>
                            {isOpen ? (
                              <ChevronUp className="w-4 h-4 text-slate-400" />
                            ) : (
                              <ChevronDown className="w-4 h-4 text-slate-400" />
                            )}
                          </button>
                          {isOpen && (
                            <div className="px-4 pb-4 border-t border-slate-100">
                              {isLoadingThis ? (
                                <div className="text-sm text-slate-400 py-4 text-center">Loading transcript...</div>
                              ) : (
                                <div className="mt-3 text-sm text-slate-700 leading-relaxed whitespace-pre-wrap max-h-[600px] overflow-y-auto">
                                  {transcriptTexts[key] || ''}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* ── 7. SEC Filings ── */}
              {secFilings.length > 0 && (
                <div className="bg-white rounded-xl border border-slate-200 p-5">
                  <h2 className="text-lg font-semibold text-[#0a1628] mb-3">SEC Filings</h2>
                  <div className="space-y-2">
                    {secFilings.map((filing, idx) => {
                      const displayTitle = `${filing.filing_type} FY${filing.fiscal_year}${
                        filing.quarter ? ` Q${filing.quarter}` : ''
                      }`
                      const displayDate = filing.filing_date
                        ? new Date(filing.filing_date).toLocaleDateString()
                        : null
                      return (
                        <div key={idx} className="border border-slate-200 rounded-lg overflow-hidden">
                          <div className="flex items-center justify-between px-4 py-3 hover:bg-slate-50">
                            <div className="flex items-center gap-3 flex-1">
                              <Briefcase className="w-4 h-4 text-slate-400" />
                              <span className="text-sm font-medium text-[#0a1628]">
                                {displayTitle}
                              </span>
                              {displayDate && (
                                <span className="text-xs text-slate-400">{displayDate}</span>
                              )}
                            </div>
                            <button
                              onClick={() => setDocumentPanel({
                                open: true,
                                content: {
                                  type: 'sec-filing',
                                  ticker: filing.ticker,
                                  filingType: filing.filing_type,
                                  fiscalYear: filing.fiscal_year,
                                  quarter: filing.quarter || undefined,
                                  filingDate: filing.filing_date || undefined,
                                }
                              })}
                              className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg transition-colors text-sm font-medium"
                            >
                              <Eye className="w-3.5 h-3.5" />
                              View Filing
                            </button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>

      {/* Document Panel (right sidebar) */}
      <DocumentPanel
        isOpen={documentPanel.open}
        onClose={() => setDocumentPanel(prev => ({ ...prev, open: false }))}
        content={documentPanel.content}
      />
    </div>
  )
}

// ── Sub-components ──

function SegmentCard({ title, segments }: { title: string; segments: Segment[] }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <h3 className="text-sm font-semibold text-[#0a1628] mb-3">{title}</h3>
      <div className="space-y-2.5">
        {segments.map((seg) => (
          <div key={seg.segment}>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-slate-600 truncate mr-2">{seg.segment}</span>
              <span className="text-[#0a1628] font-medium whitespace-nowrap">
                {seg.formatted_revenue} ({seg.formatted_percentage})
              </span>
            </div>
            <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-[#0083f1] rounded-full"
                style={{ width: `${Math.min(seg.percentage, 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

const INCOME_ROWS: [string, string][] = [
  ['revenue', 'Revenue'],
  ['costOfRevenue', 'Cost of Revenue'],
  ['grossProfit', 'Gross Profit'],
  ['operatingExpenses', 'Operating Expenses'],
  ['operatingIncome', 'Operating Income'],
  ['ebitda', 'EBITDA'],
  ['netIncome', 'Net Income'],
  ['eps', 'EPS'],
  ['epsDiluted', 'EPS (Diluted)'],
]

const BALANCE_ROWS: [string, string][] = [
  ['cashAndCashEquivalents', 'Cash & Equivalents'],
  ['totalCurrentAssets', 'Total Current Assets'],
  ['totalAssets', 'Total Assets'],
  ['totalCurrentLiabilities', 'Total Current Liabilities'],
  ['totalLiabilities', 'Total Liabilities'],
  ['totalDebt', 'Total Debt'],
  ['totalStockholdersEquity', "Stockholders' Equity"],
  ['netDebt', 'Net Debt'],
]

const CASHFLOW_ROWS: [string, string][] = [
  ['netIncome', 'Net Income'],
  ['operatingCashFlow', 'Operating Cash Flow'],
  ['capitalExpenditure', 'Capital Expenditure'],
  ['freeCashFlow', 'Free Cash Flow'],
  ['netCashUsedForInvestingActivites', 'Investing Activities'],
  ['netCashUsedProvidedByFinancingActivities', 'Financing Activities'],
  ['netChangeInCash', 'Net Change in Cash'],
]

function FinancialTable({
  tab,
  income,
  balance,
  cashflow,
}: {
  tab: 'income' | 'balance' | 'cashflow'
  income: FinancialStatement[]
  balance: FinancialStatement[]
  cashflow: FinancialStatement[]
}) {
  const rows = tab === 'income' ? INCOME_ROWS : tab === 'balance' ? BALANCE_ROWS : CASHFLOW_ROWS
  const statements = tab === 'income' ? income : tab === 'balance' ? balance : cashflow

  if (statements.length === 0) {
    return <div className="p-6 text-center text-sm text-slate-400">No data available</div>
  }

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="bg-slate-50 border-b border-slate-200">
          <th className="px-4 py-2.5 text-left text-slate-500 font-medium sticky left-0 bg-slate-50 min-w-[180px]">
            Metric
          </th>
          {statements.map((s) => (
            <th key={String(s.calendarYear)} className="px-4 py-2.5 text-right text-slate-500 font-medium whitespace-nowrap min-w-[110px]">
              FY {String(s.calendarYear)}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(([key, label]) => (
          <tr key={key} className="border-b border-slate-100 hover:bg-slate-50/50">
            <td className="px-4 py-2.5 text-slate-600 font-medium sticky left-0 bg-white">{label}</td>
            {statements.map((s) => (
              <td key={String(s.calendarYear)} className="px-4 py-2.5 text-right text-[#0a1628] whitespace-nowrap">
                {key === 'eps' || key === 'epsDiluted'
                  ? fmt(s[key], 'currency')
                  : fmt(s[key], 'currency')}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
