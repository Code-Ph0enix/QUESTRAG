import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { RiGithubFill, RiArrowRightLine, RiSparklingFill } from 'react-icons/ri';

export default function CTA() {
  return (
    <section className="relative py-20 md:py-32 overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 bg-gradient-to-br from-violet-600 via-blue-600 to-cyan-600" />
      
      {/* Animated orbs */}
      <motion.div 
        className="absolute top-0 left-1/4 w-96 h-96 bg-white/10 rounded-full blur-3xl"
        animate={{ 
          scale: [1, 1.3, 1],
          x: [0, 50, 0],
          opacity: [0.1, 0.2, 0.1]
        }}
        transition={{ duration: 10, repeat: Infinity }}
      />
      <motion.div 
        className="absolute bottom-0 right-1/4 w-80 h-80 bg-cyan-300/20 rounded-full blur-3xl"
        animate={{ 
          scale: [1, 1.4, 1],
          x: [0, -50, 0],
          opacity: [0.1, 0.25, 0.1]
        }}
        transition={{ duration: 8, repeat: Infinity, delay: 2 }}
      />
      
      {/* Grid overlay */}
      <div 
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
          backgroundSize: '50px 50px',
        }}
      />

      <div className="container relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, type: "spring" }}
          className="max-w-4xl mx-auto text-center"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/20 backdrop-blur-sm mb-8"
          >
            <RiSparklingFill className="w-4 h-4 text-yellow-300" />
            <span className="text-sm font-medium text-white/90">Open Source & Free to Use</span>
          </motion.div>
          
          {/* Heading */}
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6">
            Ready to Transform Your
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-200 via-white to-violet-200">
              Banking Experience?
            </span>
          </h2>
          
          {/* Subtitle */}
          <p className="text-lg md:text-xl text-white/80 max-w-2xl mx-auto mb-10 leading-relaxed">
            Experience the power of AI-driven banking assistance. 
            Sign up today or explore our open-source project on GitHub!
          </p>
          
          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
            className="flex flex-col sm:flex-row justify-center gap-4"
          >
            <Link to="/signup">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="group relative overflow-hidden px-8 py-4 rounded-xl bg-white text-violet-700 font-semibold text-lg shadow-2xl shadow-black/20 hover:shadow-black/30 transition-all duration-300"
              >
                <span className="relative z-10 flex items-center justify-center gap-2">
                  Sign Up Now
                  <motion.span
                    animate={{ x: [0, 5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    <RiArrowRightLine className="w-5 h-5" />
                  </motion.span>
                </span>
                {/* Hover effect */}
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-violet-100 to-cyan-100"
                  initial={{ x: '-100%' }}
                  whileHover={{ x: 0 }}
                  transition={{ duration: 0.3 }}
                />
              </motion.button>
            </Link>
            
            <a
              href="https://github.com/Code-Ph0enix/QUESTRAG"
              target="_blank"
              rel="noreferrer"
            >
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="flex items-center justify-center gap-2 px-8 py-4 rounded-xl border-2 border-white/30 bg-white/10 backdrop-blur-sm text-white font-semibold text-lg hover:bg-white/20 hover:border-white/50 transition-all duration-300"
              >
                <RiGithubFill className="w-5 h-5" />
                View on GitHub
              </motion.button>
            </a>
          </motion.div>
          
          {/* Stats */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.6 }}
            className="mt-12 flex flex-wrap justify-center gap-8 md:gap-16"
          >
            {[
              { label: 'Happy Users', value: '1K+' },
              { label: 'Accuracy', value: '99.7%' },
              { label: 'Open Source', value: '100%' },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl md:text-3xl font-bold text-white">{stat.value}</div>
                <div className="text-sm text-white/60">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
