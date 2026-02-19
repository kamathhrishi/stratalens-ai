import { Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import ChatPage from './pages/ChatPage'
import SignInPage from './pages/SignInPage'
import SignUpPage from './pages/SignUpPage'
import ScreenerPage from './pages/ScreenerPage'
import CompaniesPage from './pages/CompaniesPage'
import CompanyPage from './pages/CompanyPage'
import PortfolioPage from './pages/PortfolioPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/sign-in/*" element={<SignInPage />} />
      <Route path="/sign-up/*" element={<SignUpPage />} />
      <Route path="/chat" element={<ChatPage />} />
      <Route path="/screener" element={<ScreenerPage />} />
      <Route path="/companies" element={<CompaniesPage />} />
      <Route path="/companies/:symbol" element={<CompanyPage />} />
      <Route path="/portfolio" element={<PortfolioPage />} />
    </Routes>
  )
}

export default App
