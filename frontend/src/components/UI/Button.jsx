import { motion } from 'framer-motion'

const Button = ({ 
  children, 
  onClick, 
  variant = 'primary', 
  size = 'md',
  disabled = false,
  loading = false,
  icon,
  className = '',
  ...props 
}) => {
  const baseStyles = 'btn inline-flex items-center justify-center gap-2 font-semibold rounded-xl transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 dark:focus:ring-offset-gray-900 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl active:shadow-md'
  
  const variants = {
    primary: 'bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 hover:from-violet-700 hover:via-purple-700 hover:to-fuchsia-700 text-white focus:ring-violet-500 shadow-violet-500/30 hover:shadow-violet-500/50',
    secondary: 'bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 border-2 border-violet-200 dark:border-violet-800 focus:ring-violet-400 shadow-violet-200/50 dark:shadow-violet-900/50',
    success: 'bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white focus:ring-emerald-500 shadow-emerald-500/30',
    danger: 'bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 text-white focus:ring-red-500 shadow-red-500/30',
    ghost: 'hover:bg-violet-100 dark:hover:bg-violet-900/30 text-violet-700 dark:text-violet-300 shadow-none hover:shadow-md',
    outline: 'border-2 border-violet-500 text-violet-600 dark:text-violet-400 hover:bg-violet-50 dark:hover:bg-violet-900/30 focus:ring-violet-500 shadow-none',
  }
  
  const sizes = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-2.5 text-base',
    lg: 'px-8 py-3.5 text-lg',
  }
  
  const classes = `${baseStyles} ${variants[variant] || variants.primary} ${sizes[size]} ${className}`
  
  return (
    <motion.button
      whileHover={{ scale: disabled || loading ? 1 : 1.05 }}
      whileTap={{ scale: disabled || loading ? 1 : 0.95 }}
      onClick={onClick}
      disabled={disabled || loading}
      className={classes}
      {...props}
    >
      {loading && (
  <motion.svg 
    className="h-5 w-5" 
    xmlns="http://www.w3.org/2000/svg" 
    fill="none" 
    viewBox="0 0 24 24"
    animate={{ rotate: 360 }}
    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
  >
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
  </motion.svg>
)}

      {icon && !loading && (
        <motion.span
          whileHover={{ scale: 1.2, rotate: 15 }}
          transition={{ duration: 0.2 }}
        >
          {icon}
        </motion.span>
      )}
      <span>{children}</span>
    </motion.button>
  )
}

export default Button
