import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HiMoon, HiSun } from 'react-icons/hi';
import { RiGithubFill, RiTwitterFill, RiLinkedinFill, RiHeartFill } from 'react-icons/ri';
import { useTheme } from '../../context/ThemeContext';
import Logo from '../shared/Logo';

export default function Footer() {
  const { theme, toggleTheme } = useTheme();

  return (
    <footer className="relative z-10 w-full border-t border-border/50 py-12 bg-gradient-to-b from-background to-violet-50/30 dark:to-violet-950/20">
      <div className="container">
        <div className="grid gap-8 md:grid-cols-4">
          {/* Brand */}
          <div className="md:col-span-2">
            <Logo size="md" showText={true} animated={false} />
            <p className="mt-4 text-sm text-muted-foreground max-w-md leading-relaxed">
              Intelligent banking assistant powered by advanced RAG and reinforcement learning. 
              Get instant, accurate answers to all your banking queries.
            </p>
            {/* Social Links */}
            <div className="flex items-center gap-3 mt-6">
              {[
                { icon: RiGithubFill, href: 'https://github.com/Code-Ph0enix/QUESTRAG', label: 'GitHub' },
                { icon: RiTwitterFill, href: '#', label: 'Twitter' },
                { icon: RiLinkedinFill, href: '#', label: 'LinkedIn' },
              ].map((social, i) => (
                <motion.a
                  key={i}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.1, y: -2 }}
                  whileTap={{ scale: 0.9 }}
                  className="h-10 w-10 rounded-xl bg-gradient-to-br from-violet-100 to-cyan-100 dark:from-violet-900/30 dark:to-cyan-900/30 flex items-center justify-center border border-violet-200/50 dark:border-violet-500/20 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <social.icon className="h-5 w-5" />
                  <span className="sr-only">{social.label}</span>
                </motion.a>
              ))}
            </div>
          </div>
          
          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-foreground mb-4">Quick Links</h4>
            <ul className="space-y-3">
              {['Home', 'Features', 'About', 'Chat'].map((link) => (
                <li key={link}>
                  <Link 
                    to={link === 'Home' ? '/' : `/${link.toLowerCase()}`}
                    className="text-sm text-muted-foreground hover:text-violet-600 dark:hover:text-violet-400 transition-colors"
                  >
                    {link}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Resources */}
          <div>
            <h4 className="font-semibold text-foreground mb-4">Resources</h4>
            <ul className="space-y-3">
              <li>
                <a 
                  href="https://github.com/Code-Ph0enix/QUESTRAG"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-muted-foreground hover:text-violet-600 dark:hover:text-violet-400 transition-colors"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <a 
                  href="https://github.com/Code-Ph0enix/QUESTRAG/issues"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-muted-foreground hover:text-violet-600 dark:hover:text-violet-400 transition-colors"
                >
                  Report Issues
                </a>
              </li>
              <li>
                <a 
                  href="https://github.com/Code-Ph0enix/QUESTRAG#readme"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-muted-foreground hover:text-violet-600 dark:hover:text-violet-400 transition-colors"
                >
                  Documentation
                </a>
              </li>
            </ul>
          </div>
        </div>
        
        {/* Bottom Bar */}
        <div className="mt-12 pt-8 border-t border-border/50 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-muted-foreground flex items-center gap-1">
            Made with <RiHeartFill className="w-4 h-4 text-red-500" /> by{" "}
            <a
              href="https://github.com/Code-Ph0enix"
              target="_blank"
              rel="noreferrer"
              className="font-medium text-violet-600 dark:text-violet-400 hover:underline"
            >
              Code-Ph0enix
            </a>
          </p>
          
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">© 2024 QuestRAG. All rights reserved.</span>
            
            {/* Theme Toggle */}
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={toggleTheme}
              className="h-9 w-9 rounded-xl bg-gradient-to-br from-violet-100 to-cyan-100 dark:from-violet-900/30 dark:to-cyan-900/30 flex items-center justify-center border border-violet-200/50 dark:border-violet-500/20"
            >
              {theme === 'dark' ? (
                <HiSun className="h-4 w-4 text-amber-400" />
              ) : (
                <HiMoon className="h-4 w-4 text-violet-600" />
              )}
            </motion.button>
          </div>
        </div>
      </div>
    </footer>
  );
}
