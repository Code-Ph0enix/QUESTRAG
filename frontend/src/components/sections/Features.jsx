import { motion } from 'framer-motion';
import { 
  RiBrainLine, 
  RiDatabase2Line, 
  RiShieldCheckLine,
  RiSpeedLine,
  RiCloudLine,
  RiLockLine,
  RiCodeLine,
  RiMessageLine,
  RiSearchLine
} from 'react-icons/ri';

// Feature card with glassmorphism and hover effects
const FeatureCard = ({ feature, index }) => (
  <motion.div
    initial={{ opacity: 0, y: 40, rotateX: -10 }}
    whileInView={{ opacity: 1, y: 0, rotateX: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.6, delay: index * 0.08, type: "spring", stiffness: 100 }}
    whileHover={{ y: -10, scale: 1.02 }}
    className="group relative"
  >
    {/* Glow effect on hover */}
    <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-500`} />
    
    {/* Card */}
    <div className="relative h-full backdrop-blur-sm bg-white/70 dark:bg-gray-900/70 border border-violet-200/50 dark:border-violet-500/20 rounded-2xl p-6 transition-all duration-300 group-hover:shadow-2xl group-hover:shadow-violet-500/20 overflow-hidden">
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5 group-hover:opacity-10 transition-opacity duration-300">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 2px 2px, currentColor 1px, transparent 1px)`,
          backgroundSize: '24px 24px'
        }} />
      </div>
      
      {/* Icon container */}
      <motion.div 
        className={`relative mx-auto w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 shadow-lg`}
        whileHover={{ rotate: [0, -5, 5, 0], scale: 1.1 }}
        transition={{ duration: 0.4 }}
      >
        <feature.icon className="h-8 w-8 text-white" />
        
        {/* Floating particles */}
        <motion.div
          animate={{ 
            y: [-5, 5, -5],
            opacity: [0.5, 1, 0.5]
          }}
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute -top-1 -right-1 w-3 h-3 bg-white/80 rounded-full"
        />
      </motion.div>
      
      {/* Title */}
      <h3 className={`text-lg font-bold mb-2 bg-gradient-to-r ${feature.gradient} bg-clip-text text-transparent`}>
        {feature.title}
      </h3>
      
      {/* Description */}
      <p className="text-sm text-muted-foreground leading-relaxed relative z-10">
        {feature.description}
      </p>
      
      {/* Bottom accent line */}
      <motion.div 
        className={`absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r ${feature.gradient}`}
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8, delay: index * 0.1 + 0.3 }}
      />
    </div>
  </motion.div>
);

export default function Features() {
  return (
    <section className="relative py-20 overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-violet-50/30 dark:via-violet-950/20 to-background" />
      
      {/* Floating orbs */}
      <motion.div 
        className="absolute top-20 left-10 w-72 h-72 bg-violet-400/20 rounded-full blur-3xl"
        animate={{ 
          scale: [1, 1.2, 1],
          opacity: [0.1, 0.2, 0.1]
        }}
        transition={{ duration: 8, repeat: Infinity }}
      />
      <motion.div 
        className="absolute bottom-20 right-10 w-96 h-96 bg-cyan-400/20 rounded-full blur-3xl"
        animate={{ 
          scale: [1, 1.3, 1],
          opacity: [0.1, 0.25, 0.1]
        }}
        transition={{ duration: 10, repeat: Infinity, delay: 2 }}
      />

      <div className="container relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="mx-auto max-w-3xl text-center mb-16"
        >
          <motion.span 
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="inline-block px-4 py-1.5 mb-4 rounded-full bg-gradient-to-r from-violet-500/10 via-blue-500/10 to-cyan-500/10 border border-violet-500/20 text-sm font-medium text-violet-600 dark:text-violet-400"
          >
            ✨ Core Features
          </motion.span>
          
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6">
            <span className="bg-gradient-to-r from-violet-600 via-blue-500 to-cyan-500 bg-clip-text text-transparent">
              Powerful Capabilities
            </span>
          </h2>
          
          <p className="text-lg text-muted-foreground leading-relaxed">
            Built with cutting-edge technology to deliver 
            <span className="font-semibold text-violet-600 dark:text-violet-400"> accurate</span>, 
            <span className="font-semibold text-blue-600 dark:text-blue-400"> fast</span>, and 
            <span className="font-semibold text-cyan-600 dark:text-cyan-400"> secure</span> banking assistance.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 max-w-6xl mx-auto">
          {features.map((feature, i) => (
            <FeatureCard key={i} feature={feature} index={i} />
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, delay: 0.5 }}
          className="mt-16 text-center"
        >
          <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-gradient-to-r from-violet-500/10 via-blue-500/10 to-cyan-500/10 border border-violet-500/20 backdrop-blur-sm">
            <RiBrainLine className="w-5 h-5 text-violet-500" />
            <span className="text-sm font-medium text-foreground">
              QuestRAG combines <span className="text-violet-600 dark:text-violet-400">RAG architecture</span> with 
              <span className="text-blue-600 dark:text-blue-400"> reinforcement learning</span> for intelligent responses
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

const features = [
  {
    icon: RiBrainLine,
    title: "Neural RAG System",
    description: "Advanced RAG system with neural retrieval for deep contextual understanding of banking queries.",
    gradient: "from-violet-500 to-purple-600"
  },
  {
    icon: RiDatabase2Line,
    title: "Massive Knowledge Base",
    description: "19,352 banking documents indexed for comprehensive knowledge coverage and accurate responses.",
    gradient: "from-blue-500 to-cyan-500"
  },
  {
    icon: RiShieldCheckLine,
    title: "99.7% Accuracy",
    description: "RL-powered policy network achieving industry-leading response accuracy for banking assistance.",
    gradient: "from-violet-500 to-blue-500"
  },
  {
    icon: RiSpeedLine,
    title: "Lightning Fast",
    description: "Sub-second response times with optimized FAISS vector search and efficient retrieval.",
    gradient: "from-cyan-500 to-teal-500"
  },
  {
    icon: RiCloudLine,
    title: "Cloud Persistence",
    description: "MongoDB Atlas for secure, real-time conversation persistence across all sessions.",
    gradient: "from-blue-500 to-violet-500"
  },
  {
    icon: RiLockLine,
    title: "Enterprise Security",
    description: "End-to-end encryption with JWT authentication ensuring complete data security.",
    gradient: "from-purple-500 to-pink-500"
  },
  {
    icon: RiCodeLine,
    title: "Modern Stack",
    description: "Built with FastAPI, React, and PyTorch for a robust, scalable architecture.",
    gradient: "from-violet-500 to-cyan-500"
  },
  {
    icon: RiMessageLine,
    title: "Context Aware",
    description: "Intelligent conversation history understanding for coherent, contextual responses.",
    gradient: "from-blue-500 to-purple-500"
  },
  {
    icon: RiSearchLine,
    title: "Smart Retrieval",
    description: "Semantic search with FAISS vector indexing for finding the most relevant information.",
    gradient: "from-cyan-500 to-blue-500"
  }
];
