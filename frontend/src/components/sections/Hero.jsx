import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HiOutlineStar, HiOutlineSparkles, HiOutlineLightningBolt, HiOutlineChartBar } from 'react-icons/hi';
import { RiGithubFill, RiSparklingFill } from 'react-icons/ri';
import { getButtonClasses } from '../shadcn';
import { cn } from '../../lib/utils';
import AnimatedCounter from '../shared/AnimatedCounter';

// Floating particles component
const FloatingParticles = () => {
  const particles = Array.from({ length: 25 }, (_, i) => ({
    id: i,
    size: Math.random() * 6 + 2,
    x: Math.random() * 100,
    y: Math.random() * 100,
    duration: Math.random() * 10 + 15,
    delay: Math.random() * 5,
  }));

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full"
          style={{
            width: particle.size,
            height: particle.size,
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            background: `linear-gradient(135deg, rgba(139, 92, 246, 0.4), rgba(6, 182, 212, 0.4))`,
            filter: 'blur(1px)',
          }}
          animate={{
            y: [0, -40, 0],
            x: [0, Math.random() * 30 - 15, 0],
            opacity: [0.2, 0.6, 0.2],
            scale: [1, 1.5, 1],
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            delay: particle.delay,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
};

// Animated gradient orb
const GradientOrb = ({ className, delay = 0 }) => (
  <motion.div
    className={`absolute rounded-full blur-3xl ${className}`}
    animate={{
      scale: [1, 1.3, 1],
      opacity: [0.15, 0.35, 0.15],
      rotate: [0, 180, 360],
    }}
    transition={{
      duration: 12,
      repeat: Infinity,
      delay,
      ease: "easeInOut",
    }}
  />
);

// Stats card with glass morphism
const StatCard = ({ icon: Icon, value, suffix, prefix, label, delay, decimals = 0 }) => (
  <motion.div
    initial={{ opacity: 0, y: 40, scale: 0.8 }}
    whileInView={{ opacity: 1, y: 0, scale: 1 }}
    viewport={{ once: true }}
    transition={{ duration: 0.7, delay, type: "spring", stiffness: 100 }}
    whileHover={{ y: -8, scale: 1.05 }}
    className="relative group cursor-pointer"
  >
    {/* Glow effect */}
    <div className="absolute inset-0 bg-gradient-to-r from-violet-500/30 via-blue-500/30 to-cyan-500/30 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
    
    {/* Card */}
    <div className="relative backdrop-blur-xl bg-white/80 dark:bg-gray-900/80 border border-violet-200/50 dark:border-violet-500/20 rounded-2xl p-6 text-center shadow-xl shadow-violet-500/10">
      {/* Icon */}
      <motion.div 
        className="mx-auto w-14 h-14 rounded-2xl bg-gradient-to-br from-violet-500 via-blue-500 to-cyan-500 flex items-center justify-center mb-4 shadow-lg shadow-violet-500/30"
        whileHover={{ rotate: [0, -10, 10, 0] }}
        transition={{ duration: 0.5 }}
      >
        <Icon className="w-7 h-7 text-white" />
      </motion.div>
      
      {/* Value */}
      <div className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-violet-600 via-blue-500 to-cyan-500 bg-clip-text text-transparent">
        {prefix}<AnimatedCounter target={value} suffix={suffix} decimals={decimals} duration={2500} />
      </div>
      
      {/* Label */}
      <p className="text-sm text-muted-foreground mt-2 font-medium">{label}</p>
      
      {/* Decorative line */}
      <motion.div 
        className="absolute bottom-0 left-1/2 -translate-x-1/2 h-1 bg-gradient-to-r from-violet-500 via-blue-500 to-cyan-500 rounded-full"
        initial={{ width: 0 }}
        whileInView={{ width: '60%' }}
        viewport={{ once: true }}
        transition={{ duration: 0.8, delay: delay + 0.3 }}
      />
    </div>
  </motion.div>
);

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 bg-gradient-to-br from-violet-100/50 via-background to-cyan-100/50 dark:from-violet-950/30 dark:via-background dark:to-cyan-950/30" />
      
      {/* Gradient orbs */}
      <GradientOrb className="w-[800px] h-[800px] bg-violet-500 -top-96 -left-48" delay={0} />
      <GradientOrb className="w-[600px] h-[600px] bg-blue-500 top-1/3 -right-48" delay={3} />
      <GradientOrb className="w-[500px] h-[500px] bg-cyan-500 -bottom-48 left-1/4" delay={6} />

      {/* Floating particles */}
      <FloatingParticles />

      {/* Animated grid */}
      <div 
        className="absolute inset-0 opacity-[0.03] dark:opacity-[0.07]"
        style={{
          backgroundImage: `linear-gradient(rgba(139, 92, 246, 0.8) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(6, 182, 212, 0.8) 1px, transparent 1px)`,
          backgroundSize: '60px 60px',
        }}
      />

      <div className="container relative z-10 py-20 md:py-28">
        <div className="mx-auto w-full max-w-4xl text-center">
          {/* Announcement Banner */}
          <motion.a
            href="https://github.com/Code-Ph0enix/QUESTRAG"
            title="View on GitHub"
            target="_blank"
            rel="noreferrer"
            initial={{ opacity: 0, y: -30, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.6, type: "spring" }}
            whileHover={{ scale: 1.05 }}
            className="group mx-auto mb-8 inline-flex items-center justify-center gap-2 overflow-hidden rounded-full bg-gradient-to-r from-violet-100 via-blue-100 to-cyan-100 dark:from-violet-900/40 dark:via-blue-900/40 dark:to-cyan-900/40 px-6 py-2.5 border border-violet-200/50 dark:border-violet-500/30 backdrop-blur-sm transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/20"
          >
            <motion.span
              animate={{ rotate: [0, 15, -15, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <RiSparklingFill className="h-5 w-5 text-violet-600 dark:text-violet-400" />
            </motion.span>
            <span className="text-sm font-semibold bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-600 bg-clip-text text-transparent">
              Introducing QuestRAG AI Assistant
            </span>
            <motion.span
              animate={{ x: [0, 4, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="text-violet-500"
            >
              →
            </motion.span>
          </motion.a>

          {/* Main Heading */}
          <motion.h1
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1, type: "spring" }}
            className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight mb-6"
          >
            <span className="text-foreground">Intelligent Banking</span>
            <br />
            <span className="relative inline-block mt-2">
              <span className="bg-gradient-to-r from-violet-600 via-blue-500 to-cyan-500 bg-clip-text text-transparent">
                Assistant
              </span>
              {/* Animated underline */}
              <motion.span
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 1, delay: 0.8 }}
                className="absolute -bottom-2 left-0 right-0 h-1.5 bg-gradient-to-r from-violet-600 via-blue-500 to-cyan-500 rounded-full origin-left"
              />
            </span>
            <span className="text-foreground"> Powered by RAG</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="mt-6 text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed"
          >
            Advanced retrieval-augmented generation with reinforcement learning for 
            <span className="font-semibold text-violet-600 dark:text-violet-400"> accurate</span>, 
            <span className="font-semibold text-blue-600 dark:text-blue-400"> contextual</span>, and 
            <span className="font-semibold text-cyan-600 dark:text-cyan-400"> intelligent</span> banking assistance.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link to="/login">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="group relative overflow-hidden px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-600 text-white font-semibold text-lg shadow-xl shadow-violet-500/30 hover:shadow-2xl hover:shadow-violet-500/40 transition-all duration-300"
              >
                <span className="relative z-10 flex items-center gap-2">
                  Get Started
                  <motion.span
                    animate={{ x: [0, 5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    →
                  </motion.span>
                </span>
                {/* Shine effect */}
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/25 to-transparent -skew-x-12"
                  animate={{ x: ['-200%', '200%'] }}
                  transition={{ duration: 3, repeat: Infinity, repeatDelay: 3 }}
                />
              </motion.button>
            </Link>
            
            <a
              href="https://github.com/Code-Ph0enix/QUESTRAG"
              target="_blank"
              rel="noopener noreferrer"
            >
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="flex items-center gap-2 px-8 py-4 rounded-xl border-2 border-violet-300 dark:border-violet-600 bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm font-semibold text-lg text-foreground hover:border-violet-500 hover:bg-violet-50 dark:hover:bg-violet-900/30 transition-all duration-300"
              >
                <HiOutlineStar className="h-5 w-5 text-violet-500" />
                <span>Star on</span>
                <RiGithubFill className="h-5 w-5" />
              </motion.button>
            </a>
          </motion.div>
        </div>

        {/* Stats Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-24"
        >
          <motion.h2 
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="mb-10 text-center text-2xl md:text-3xl font-bold bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-600 bg-clip-text text-transparent"
          >
            Powered by Advanced Technology
          </motion.h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            <StatCard
              icon={HiOutlineChartBar}
              value={19000}
              suffix="+"
              prefix=""
              label="Documents Processed"
              delay={0.5}
              decimals={0}
            />
            <StatCard
              icon={HiOutlineSparkles}
              value={99.7}
              suffix="%"
              prefix=""
              label="Accuracy Rate"
              delay={0.6}
              decimals={1}
            />
            <StatCard
              icon={HiOutlineLightningBolt}
              value={1}
              suffix="s"
              prefix="<"
              label="Avg Response Time"
              delay={0.7}
              decimals={0}
            />
          </div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-6 h-10 rounded-full border-2 border-violet-400/50 flex justify-center pt-2"
          >
            <motion.div
              animate={{ opacity: [0, 1, 0], y: [0, 12, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-1.5 h-2.5 bg-gradient-to-b from-violet-500 to-cyan-500 rounded-full"
            />
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
