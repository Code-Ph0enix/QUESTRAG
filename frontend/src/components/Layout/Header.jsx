import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { HiMenuAlt2, HiX, HiMoon, HiSun, HiChat, HiLogout, HiCog, HiChevronDown } from 'react-icons/hi';
import { useTheme } from '../../context/ThemeContext';
import { useAuth } from '../../context/AuthContext';
import { Button, getButtonClasses } from '../shadcn';
import { cn } from '../../lib/utils';
import Logo from '../shared/Logo';

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    setShowUserMenu(false);
    navigate('/');
  };

  return (
    <header className="h-20 w-full border-b border-border/50 bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
      <div className="container h-full">
        <nav className="flex h-full items-center justify-between">
          {/* Logo */}
          <Link to="/">
            <Logo size="md" showText={true} animated={true} />
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden items-center gap-12 lg:flex 2xl:gap-16">
            <div className="flex items-center gap-8 text-sm">
              {['About', 'Features'].map((item) => (
                <Link
                  key={item}
                  to={`/${item.toLowerCase()}`}
                  className="relative font-medium text-muted-foreground hover:text-foreground transition-colors group"
                >
                  {item}
                  <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-violet-500 to-cyan-500 group-hover:w-full transition-all duration-300" />
                </Link>
              ))}
              <a
                href="https://github.com/Code-Ph0enix/QUESTRAG"
                target="_blank"
                rel="noopener noreferrer"
                className="relative font-medium text-muted-foreground hover:text-foreground transition-colors group"
              >
                GitHub
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-violet-500 to-cyan-500 group-hover:w-full transition-all duration-300" />
              </a>
            </div>
            <div className="flex items-center gap-x-3">
              {/* Theme Toggle */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={toggleTheme}
                className="relative h-10 w-10 rounded-xl bg-gradient-to-br from-violet-100 to-cyan-100 dark:from-violet-900/30 dark:to-cyan-900/30 flex items-center justify-center border border-violet-200/50 dark:border-violet-500/20"
              >
                {theme === 'dark' ? (
                  <HiSun className="h-5 w-5 text-yellow-500" />
                ) : (
                  <HiMoon className="h-5 w-5 text-violet-600" />
                )}
              </motion.button>

              {isAuthenticated ? (
                <>
                  {/* Go to Chat Button */}
                  <Link to="/chat">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-600 text-white font-medium shadow-lg shadow-violet-500/25 hover:shadow-xl hover:shadow-violet-500/30 transition-all flex items-center gap-2"
                    >
                      <HiChat className="w-4 h-4" />
                      Go to Chat
                    </motion.button>
                  </Link>

                  {/* User Profile Menu */}
                  <div className="relative">
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      onClick={() => setShowUserMenu(!showUserMenu)}
                      className="flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-accent transition-colors"
                    >
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center text-white text-sm font-medium">
                        {user?.full_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
                      </div>
                      <span className="text-sm font-medium hidden xl:block">
                        {user?.full_name || 'User'}
                      </span>
                      <HiChevronDown className={cn(
                        "w-4 h-4 transition-transform",
                        showUserMenu && "rotate-180"
                      )} />
                    </motion.button>

                    <AnimatePresence>
                      {showUserMenu && (
                        <motion.div
                          initial={{ opacity: 0, y: 10, scale: 0.95 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, y: 10, scale: 0.95 }}
                          className="absolute right-0 top-full mt-2 w-56 bg-popover border border-border rounded-xl shadow-xl overflow-hidden z-50"
                        >
                          <div className="px-4 py-3 border-b border-border">
                            <p className="text-sm font-medium">{user?.full_name || 'User'}</p>
                            <p className="text-xs text-muted-foreground">{user?.email}</p>
                          </div>
                          <div className="py-1">
                            <Link
                              to="/chat"
                              onClick={() => setShowUserMenu(false)}
                              className="flex items-center gap-3 px-4 py-2 text-sm hover:bg-accent transition-colors"
                            >
                              <HiChat className="w-4 h-4" />
                              Chat
                            </Link>
                            <button
                              onClick={() => setShowUserMenu(false)}
                              className="w-full flex items-center gap-3 px-4 py-2 text-sm hover:bg-accent transition-colors"
                            >
                              <HiCog className="w-4 h-4" />
                              Settings
                            </button>
                          </div>
                          <div className="border-t border-border py-1">
                            <button
                              onClick={handleLogout}
                              className="w-full flex items-center gap-3 px-4 py-2 text-sm text-red-500 hover:bg-accent transition-colors"
                            >
                              <HiLogout className="w-4 h-4" />
                              Sign Out
                            </button>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </>
              ) : (
                <>
                  <Link to="/login">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="px-5 py-2.5 rounded-xl border-2 border-violet-300 dark:border-violet-600 font-medium text-foreground hover:border-violet-500 hover:bg-violet-50 dark:hover:bg-violet-900/30 transition-all"
                    >
                      Sign In
                    </motion.button>
                  </Link>
                  <Link to="/signup">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-600 text-white font-medium shadow-lg shadow-violet-500/25 hover:shadow-xl hover:shadow-violet-500/30 transition-all"
                    >
                      Get Started
                    </motion.button>
                  </Link>
                </>
              )}
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex items-center gap-2 lg:hidden">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={toggleTheme}
              className="h-10 w-10 rounded-xl bg-gradient-to-br from-violet-100 to-cyan-100 dark:from-violet-900/30 dark:to-cyan-900/30 flex items-center justify-center border border-violet-200/50 dark:border-violet-500/20"
            >
              {theme === 'dark' ? (
                <HiSun className="h-5 w-5 text-amber-400" />
              ) : (
                <HiMoon className="h-5 w-5 text-violet-600" />
              )}
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="h-10 w-10 rounded-xl bg-gradient-to-br from-violet-100 to-cyan-100 dark:from-violet-900/30 dark:to-cyan-900/30 flex items-center justify-center border border-violet-200/50 dark:border-violet-500/20"
            >
              {isMenuOpen ? <HiX className="h-5 w-5" /> : <HiMenuAlt2 className="h-5 w-5" />}
            </motion.button>
          </div>
        </nav>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden border-t border-border bg-background"
          >
            <div className="container py-6 flex flex-col items-center space-y-6">
              <Link
                to="/about"
                className="font-semibold text-muted-foreground hover:text-foreground transition-colors"
                onClick={() => setIsMenuOpen(false)}
              >
                About
              </Link>
              <Link
                to="/features"
                className="font-semibold text-muted-foreground hover:text-foreground transition-colors"
                onClick={() => setIsMenuOpen(false)}
              >
                Features
              </Link>
              <a
                href="https://github.com/Code-Ph0enix/QUESTRAG"
                target="_blank"
                rel="noopener noreferrer"
                className="font-semibold text-muted-foreground hover:text-foreground transition-colors"
                onClick={() => setIsMenuOpen(false)}
              >
                GitHub
              </a>
              <div className="flex flex-col w-full gap-3 pt-4">
                {isAuthenticated ? (
                  <>
                    {/* User Info */}
                    <div className="flex items-center gap-3 px-4 py-3 bg-accent rounded-xl">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center text-white font-medium">
                        {user?.full_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
                      </div>
                      <div>
                        <p className="font-medium">{user?.full_name || 'User'}</p>
                        <p className="text-xs text-muted-foreground">{user?.email}</p>
                      </div>
                    </div>
                    <Link
                      to="/chat"
                      className={cn(getButtonClasses(), "w-full justify-center")}
                      onClick={() => setIsMenuOpen(false)}
                    >
                      <HiChat className="w-4 h-4 mr-2" />
                      Go to Chat
                    </Link>
                    <button
                      onClick={() => {
                        handleLogout();
                        setIsMenuOpen(false);
                      }}
                      className={cn(getButtonClasses("outline"), "w-full justify-center text-red-500 border-red-500/30 hover:bg-red-500/10")}
                    >
                      <HiLogout className="w-4 h-4 mr-2" />
                      Sign Out
                    </button>
                  </>
                ) : (
                  <>
                    <Link
                      to="/login"
                      className={cn(getButtonClasses("outline"), "w-full justify-center")}
                      onClick={() => setIsMenuOpen(false)}
                    >
                      Sign In
                    </Link>
                    <Link
                      to="/signup"
                      className={cn(getButtonClasses(), "w-full justify-center")}
                      onClick={() => setIsMenuOpen(false)}
                    >
                      Get Started
                    </Link>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
