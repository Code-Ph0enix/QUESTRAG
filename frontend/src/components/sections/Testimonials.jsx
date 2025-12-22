import { motion } from 'framer-motion';
import { RiDoubleQuotesL, RiStarFill } from 'react-icons/ri';
import { HiOutlineUser } from 'react-icons/hi';

const testimonials = [
  {
    name: "Banking Professional",
    title: "Financial Advisor",
    quote: "QuestRAG has transformed how I assist clients with their banking queries. The accuracy is remarkable and saves hours of research.",
    rating: 5,
    gradient: "from-violet-500 to-purple-600"
  },
  {
    name: "Tech Lead",
    title: "FinTech Startup",
    quote: "The RAG + RL architecture delivers incredibly relevant responses. A game-changer for banking AI applications.",
    rating: 5,
    gradient: "from-blue-500 to-cyan-500"
  },
  {
    name: "Customer Service Manager",
    title: "Regional Bank",
    quote: "Our response times have improved dramatically. Customers love the instant, accurate assistance they receive.",
    rating: 5,
    gradient: "from-violet-500 to-blue-500"
  },
];

const TestimonialCard = ({ testimonial, index }) => (
  <motion.div
    initial={{ opacity: 0, y: 40, scale: 0.9 }}
    whileInView={{ opacity: 1, y: 0, scale: 1 }}
    viewport={{ once: true }}
    transition={{ duration: 0.6, delay: index * 0.15, type: "spring", stiffness: 100 }}
    whileHover={{ y: -10, scale: 1.02 }}
    className="group relative"
  >
    {/* Glow effect */}
    <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${testimonial.gradient} opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-500`} />
    
    {/* Card */}
    <div className="relative h-full backdrop-blur-sm bg-white/80 dark:bg-gray-900/80 border border-violet-200/50 dark:border-violet-500/20 rounded-2xl p-6 transition-all duration-300 group-hover:shadow-2xl group-hover:shadow-violet-500/20 overflow-hidden">
      {/* Quote icon */}
      <div className={`absolute top-4 right-4 opacity-10 group-hover:opacity-20 transition-opacity`}>
        <RiDoubleQuotesL className="w-16 h-16 text-violet-500" />
      </div>
      
      {/* Rating */}
      <div className="flex gap-1 mb-4">
        {[...Array(testimonial.rating)].map((_, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, scale: 0 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.1 + i * 0.1 }}
          >
            <RiStarFill className="w-5 h-5 text-yellow-400" />
          </motion.div>
        ))}
      </div>
      
      {/* Quote */}
      <p className="text-muted-foreground leading-relaxed mb-6 relative z-10">
        "{testimonial.quote}"
      </p>
      
      {/* Author */}
      <div className="flex items-center gap-4">
        <motion.div 
          className={`h-12 w-12 rounded-xl bg-gradient-to-br ${testimonial.gradient} flex items-center justify-center shadow-lg`}
          whileHover={{ rotate: [0, -5, 5, 0] }}
          transition={{ duration: 0.4 }}
        >
          <HiOutlineUser className="h-6 w-6 text-white" />
        </motion.div>
        <div>
          <h4 className={`font-semibold bg-gradient-to-r ${testimonial.gradient} bg-clip-text text-transparent`}>
            {testimonial.name}
          </h4>
          <p className="text-sm text-muted-foreground">{testimonial.title}</p>
        </div>
      </div>
      
      {/* Bottom accent */}
      <motion.div 
        className={`absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r ${testimonial.gradient}`}
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8, delay: index * 0.15 + 0.3 }}
      />
    </div>
  </motion.div>
);

export default function Testimonials() {
  return (
    <section className="relative py-20 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-violet-50/20 dark:via-violet-950/10 to-background" />
      
      {/* Floating orbs */}
      <motion.div 
        className="absolute top-1/4 left-10 w-64 h-64 bg-violet-400/10 rounded-full blur-3xl"
        animate={{ scale: [1, 1.2, 1], opacity: [0.1, 0.2, 0.1] }}
        transition={{ duration: 8, repeat: Infinity }}
      />
      <motion.div 
        className="absolute bottom-1/4 right-10 w-80 h-80 bg-cyan-400/10 rounded-full blur-3xl"
        animate={{ scale: [1, 1.3, 1], opacity: [0.1, 0.2, 0.1] }}
        transition={{ duration: 10, repeat: Infinity, delay: 2 }}
      />

      <div className="container relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="text-center mb-16"
        >
          <motion.span 
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="inline-block px-4 py-1.5 mb-4 rounded-full bg-gradient-to-r from-violet-500/10 via-blue-500/10 to-cyan-500/10 border border-violet-500/20 text-sm font-medium text-violet-600 dark:text-violet-400"
          >
            💬 Testimonials
          </motion.span>
          
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-violet-600 via-blue-500 to-cyan-500 bg-clip-text text-transparent">
              What Our Users Say
            </span>
          </h2>
          
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Trusted by banking professionals and developers worldwide
          </p>
        </motion.div>

        {/* Testimonials Grid */}
        <div className="grid gap-6 md:grid-cols-3 max-w-5xl mx-auto">
          {testimonials.map((testimonial, index) => (
            <TestimonialCard key={index} testimonial={testimonial} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
}
