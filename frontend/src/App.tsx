import { useState, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import ChatPage from './pages/ChatPage'
import ScreenerPage from './pages/ScreenerPage'
import CompaniesPage from './pages/CompaniesPage'
import CompanyPage from './pages/CompanyPage'
import PortfolioPage from './pages/PortfolioPage'
import StrataLensLogo from './components/StrataLensLogo'

function MobileWarningModal({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-6">
      <div className="bg-white rounded-2xl shadow-2xl max-w-sm w-full p-8 flex flex-col items-center text-center gap-5">
        <div className="w-14 h-14 rounded-xl bg-[#0a1628] flex items-center justify-center">
          <StrataLensLogo size={28} className="text-white" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-[#0a1628] mb-2">Desktop Only</h2>
          <p className="text-slate-500 text-sm leading-relaxed">
            StrataLens isn't supported on mobile yet. For the best experience, open it on a desktop.
          </p>
        </div>
        <button
          onClick={onDismiss}
          className="w-full py-2.5 rounded-lg bg-[#0a1628] text-white text-sm font-medium hover:bg-[#1a2d4a] transition-colors"
        >
          Continue anyway
        </button>
      </div>
    </div>
  )
}

function App() {
  const [showMobileWarning, setShowMobileWarning] = useState(false)

  useEffect(() => {
    const isTouchDevice = navigator.maxTouchPoints > 0 || 'ontouchstart' in window
    const isMobileWidth = window.innerWidth < 1024
    if (isTouchDevice || isMobileWidth) setShowMobileWarning(true)
  }, [])

  return (
    <>
      {showMobileWarning && (
        <MobileWarningModal onDismiss={() => setShowMobileWarning(false)} />
      )}
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/sign-in/*" element={<Navigate to="/" replace />} />
        <Route path="/sign-up/*" element={<Navigate to="/" replace />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/screener" element={<ScreenerPage />} />
        <Route path="/companies" element={<CompaniesPage />} />
        <Route path="/companies/:symbol" element={<CompanyPage />} />
        <Route path="/portfolio" element={<PortfolioPage />} />
      </Routes>
    </>
  )
}

export default App
