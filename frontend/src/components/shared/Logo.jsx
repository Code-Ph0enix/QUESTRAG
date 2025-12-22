import { motion } from 'framer-motion';
import { cn } from '../../lib/utils';

export default function Logo({ size = 'md', showText = true, animated = true, className }) {
  const sizes = {
    sm: { icon: 'w-8 h-8', text: 'text-lg', tagline: 'text-[10px]' },
    md: { icon: 'w-10 h-10', text: 'text-xl', tagline: 'text-xs' },
    lg: { icon: 'w-12 h-12', text: 'text-2xl', tagline: 'text-xs' },
    xl: { icon: 'w-16 h-16', text: 'text-3xl', tagline: 'text-sm' },
  };

  const currentSize = sizes[size] || sizes.md;

  const containerVariants = {
    initial: { scale: 1 },
    hover: { scale: 1.02 },
  };

  const iconVariants = {
    initial: { rotate: 0 },
    hover: { rotate: [-2, 2, -1, 1, 0], transition: { duration: 0.4 } },
  };

  const pulseVariants = {
    initial: { opacity: 0.5, scale: 0.8 },
    animate: {
      opacity: [0.3, 0.6, 0.3],
      scale: [0.85, 1, 0.85],
      transition: { duration: 3, repeat: Infinity, ease: "easeInOut" }
    }
  };

  const nodeVariants = {
    animate: (i) => ({
      opacity: [0.4, 1, 0.4],
      scale: [0.8, 1.2, 0.8],
      transition: { 
        duration: 2, 
        repeat: Infinity, 
        delay: i * 0.3,
        ease: "easeInOut" 
      }
    })
  };

  return (
    <motion.div 
      className={cn("flex items-center gap-3 select-none cursor-pointer", className)}
      variants={containerVariants}
      initial="initial"
      whileHover="hover"
    >
      {/* Logo Icon */}
      <motion.div 
        className={cn("relative", currentSize.icon)}
        variants={iconVariants}
      >
        {/* Ambient glow */}
        {animated && (
          <motion.div
            variants={pulseVariants}
            initial="initial"
            animate="animate"
            className="absolute inset-[-4px] rounded-2xl bg-gradient-to-br from-violet-500/40 via-blue-500/40 to-cyan-500/40 blur-xl"
          />
        )}

        {/* Main container */}
        <div className={cn(
          "relative rounded-xl overflow-hidden",
          "bg-gradient-to-br from-violet-600 via-blue-600 to-cyan-500",
          "shadow-lg shadow-violet-500/25",
          "ring-1 ring-white/20",
          currentSize.icon
        )}>
          {/* Glass overlay */}
          <div className="absolute inset-0 bg-gradient-to-br from-white/25 via-transparent to-transparent" />
          
          {/* Inner pattern - neural network dots */}
          <div className="absolute inset-0 flex items-center justify-center">
            <svg 
              viewBox="0 0 40 40" 
              className="w-full h-full p-1.5"
              fill="none"
            >
              {/* Connection lines */}
              <g stroke="rgba(255,255,255,0.2)" strokeWidth="0.75">
                <line x1="20" y1="8" x2="10" y2="18" />
                <line x1="20" y1="8" x2="30" y2="18" />
                <line x1="10" y1="18" x2="20" y2="28" />
                <line x1="30" y1="18" x2="20" y2="28" />
                <line x1="10" y1="18" x2="30" y2="18" />
                <line x1="20" y1="8" x2="20" y2="28" />
              </g>
              
              {/* Neural nodes */}
              {animated ? (
                <>
                  <motion.circle 
                    cx="20" cy="8" r="2.5" 
                    fill="white"
                    custom={0}
                    variants={nodeVariants}
                    animate="animate"
                  />
                  <motion.circle 
                    cx="10" cy="18" r="2" 
                    fill="rgba(255,255,255,0.8)"
                    custom={1}
                    variants={nodeVariants}
                    animate="animate"
                  />
                  <motion.circle 
                    cx="30" cy="18" r="2" 
                    fill="rgba(255,255,255,0.8)"
                    custom={2}
                    variants={nodeVariants}
                    animate="animate"
                  />
                  <motion.circle 
                    cx="20" cy="28" r="2.5" 
                    fill="white"
                    custom={3}
                    variants={nodeVariants}
                    animate="animate"
                  />
                </>
              ) : (
                <>
                  <circle cx="20" cy="8" r="2.5" fill="white" />
                  <circle cx="10" cy="18" r="2" fill="rgba(255,255,255,0.8)" />
                  <circle cx="30" cy="18" r="2" fill="rgba(255,255,255,0.8)" />
                  <circle cx="20" cy="28" r="2.5" fill="white" />
                </>
              )}
              
              {/* Center Q letter */}
              <text 
                x="20" 
                y="22" 
                textAnchor="middle" 
                fill="white" 
                fontSize="14" 
                fontWeight="700"
                fontFamily="system-ui, -apple-system, sans-serif"
                style={{ textShadow: '0 1px 2px rgba(0,0,0,0.2)' }}
              >
                Q
              </text>
            </svg>
          </div>

          {/* Shine effect */}
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -skew-x-12"
            initial={{ x: '-200%' }}
            whileHover={{ x: '200%' }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          />
        </div>

        {/* Status indicator */}
        {animated && (
          <motion.div
            className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-400 border-2 border-background"
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [1, 0.8, 1]
            }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        )}
      </motion.div>

      {/* Text */}
      {showText && (
        <motion.div 
          className="flex flex-col"
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1, duration: 0.3 }}
        >
          <div className="flex items-baseline gap-0.5">
            <span className={cn(
              currentSize.text,
              "font-bold tracking-tight bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-500 bg-clip-text text-transparent"
            )}>
              Quest
            </span>
            <span className={cn(
              currentSize.text,
              "font-extrabold tracking-tight bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent"
            )}>
              RAG
            </span>
          </div>
          {(size === 'lg' || size === 'xl') && (
            <span className={cn(
              currentSize.tagline,
              "text-muted-foreground tracking-wide uppercase font-medium"
            )}>
              AI Banking Assistant
            </span>
          )}
        </motion.div>
      )}
    </motion.div>
  );
}
