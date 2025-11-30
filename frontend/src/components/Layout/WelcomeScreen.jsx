// import { motion } from 'framer-motion'
// import { BsRobot } from 'react-icons/bs'
// import { HiSparkles, HiLightningBolt, HiDatabase } from 'react-icons/hi'
// import Button from '../UI/Button'

// const WelcomeScreen = ({ onNewChat }) => {
//   const features = [
//     {
//       icon: <HiSparkles className="w-6 h-6" />,
//       title: 'RAG-Powered',
//       description: 'Retrieval-augmented generation for accurate banking information'
//     },
//     {
//       icon: <HiLightningBolt className="w-6 h-6" />,
//       title: 'RL-Enhanced',
//       description: 'Smart policy network decides when to fetch documents'
//     },
//     {
//       icon: <HiDatabase className="w-6 h-6" />,
//       title: 'Persistent Storage',
//       description: 'Conversations stored in MongoDB Atlas cloud database'
//     }
//   ]

//   const exampleQueries = [
//     'What is my account balance?',
//     'How do I open a savings account?',
//     'What are the interest rates for fixed deposits?',
//     'How can I apply for a credit card?'
//   ]

//   return (
//     <div className="flex flex-col items-center justify-center min-h-screen p-4 sm:p-8 bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-gray-900 dark:via-slate-900 dark:to-indigo-950">
//       <motion.div
//         initial={{ scale: 0.9, opacity: 0 }}
//         animate={{ scale: 1, opacity: 1 }}
//         transition={{ duration: 0.6, ease: "easeOut" }}
//         className="max-w-5xl w-full space-y-12"
//       >
//         {/* Logo & Title */}
//         <div className="space-y-6 text-center">
//           <motion.div
//             animate={{ 
//               rotate: [0, 10, -10, 0],
//               scale: [1, 1.05, 1]
//             }}
//             transition={{ 
//               duration: 3, 
//               repeat: Infinity, 
//               repeatDelay: 2,
//               ease: "easeInOut"
//             }}
//             className="inline-block"
//           >
//             <div className="relative">
//               {/* Glow effect */}
//               <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full blur-2xl opacity-20 animate-pulse" />
//               <BsRobot className="relative w-24 h-24 mx-auto text-transparent bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 bg-clip-text" 
//                 style={{ filter: 'drop-shadow(0 4px 12px rgba(99, 102, 241, 0.3))' }}
//               />
//             </div>
//           </motion.div>
          
//           <div className="space-y-3">
//             <h1 className="text-5xl sm:text-6xl md:text-7xl font-bold">
//               <span className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
//                 QuestRAG
//               </span>
//             </h1>
//             <p className="text-xl sm:text-2xl font-medium text-transparent bg-gradient-to-r from-slate-600 to-slate-800 dark:from-slate-300 dark:to-slate-400 bg-clip-text">
//               Your intelligent banking companion powered by AI
//             </p>
//           </div>
//         </div>

//         {/* Features - Modern Cards */}
//         <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
//           {features.map((feature, index) => (
//             <motion.div
//               key={index}
//               initial={{ y: 30, opacity: 0 }}
//               animate={{ y: 0, opacity: 1 }}
//               transition={{ 
//                 delay: index * 0.15 + 0.3,
//                 duration: 0.5,
//                 ease: "easeOut"
//               }}
//               whileHover={{ 
//                 y: -8,
//                 transition: { duration: 0.2 }
//               }}
//               className="group relative"
//             >
//               {/* Card glow on hover */}
//               <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl opacity-0 group-hover:opacity-10 blur-xl transition-opacity duration-300" />
              
//               {/* Card */}
//               <div className="relative p-8 rounded-2xl bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200 dark:border-slate-700 shadow-xl shadow-slate-200/50 dark:shadow-slate-900/50 hover:shadow-2xl hover:shadow-indigo-200/50 dark:hover:shadow-indigo-900/30 transition-all duration-300">
//                 <div className="flex flex-col items-center gap-4 text-center">
//                   {/* Icon */}
//                   <motion.div 
//                     whileHover={{ rotate: 360, scale: 1.1 }}
//                     transition={{ duration: 0.6 }}
//                     className="p-4 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/30 group-hover:shadow-blue-500/50 transition-shadow duration-300"
//                   >
//                     {feature.icon}
//                   </motion.div>
                  
//                   {/* Title */}
//                   <h3 className="text-lg font-bold bg-gradient-to-r from-slate-800 to-slate-900 dark:from-white dark:to-slate-200 bg-clip-text text-transparent">
//                     {feature.title}
//                   </h3>
                  
//                   {/* Description */}
//                   <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">
//                     {feature.description}
//                   </p>
//                 </div>
//               </div>
//             </motion.div>
//           ))}
//         </div>

//         {/* Example Queries */}
//         <div className="space-y-6">
//           <motion.h3 
//             initial={{ opacity: 0 }}
//             animate={{ opacity: 1 }}
//             transition={{ delay: 0.8 }}
//             className="text-xl font-bold text-center bg-gradient-to-r from-slate-700 to-slate-900 dark:from-slate-200 dark:to-white bg-clip-text text-transparent"
//           >
//             Try asking:
//           </motion.h3>
          
//           <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
//             {exampleQueries.map((query, index) => (
//               <motion.button
//                 key={index}
//                 initial={{ x: -20, opacity: 0 }}
//                 animate={{ x: 0, opacity: 1 }}
//                 transition={{ 
//                   delay: index * 0.1 + 0.9,
//                   duration: 0.4,
//                   ease: "easeOut"
//                 }}
//                 whileHover={{ 
//                   scale: 1.03,
//                   x: 4
//                 }}
//                 whileTap={{ scale: 0.98 }}
//                 onClick={onNewChat}
//                 className="group relative p-4 text-left rounded-xl bg-white/60 dark:bg-slate-800/60 backdrop-blur-sm border border-slate-200 dark:border-slate-700 hover:border-indigo-300 dark:hover:border-indigo-700 shadow-md hover:shadow-xl hover:shadow-indigo-200/50 dark:hover:shadow-indigo-900/30 transition-all duration-300"
//               >
//                 <div className="flex items-center gap-3">
//                   <div className="flex-shrink-0 w-2 h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 group-hover:scale-125 transition-transform duration-300" />
//                   <span className="text-sm font-medium text-slate-700 dark:text-slate-300 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors duration-300">
//                     {query}
//                   </span>
//                 </div>
//               </motion.button>
//             ))}
//           </div>
//         </div>

//         {/* CTA Button */}
//         <motion.div
//           initial={{ y: 30, opacity: 0 }}
//           animate={{ y: 0, opacity: 1 }}
//           transition={{ delay: 1.2, duration: 0.5 }}
//           className="flex justify-center"
//         >
//           <Button
//             variant="primary"
//             size="lg"
//             onClick={onNewChat}
//             className="shadow-2xl shadow-indigo-500/50 hover:shadow-indigo-500/70 font-semibold px-10 py-4 text-lg"
//           >
//             Start Chatting
//           </Button>
//         </motion.div>

//         {/* Footer Badge */}
//         <motion.div
//           initial={{ opacity: 0 }}
//           animate={{ opacity: 1 }}
//           transition={{ delay: 1.4 }}
//           className="text-center"
//         >
//           <p className="text-xs text-slate-500 dark:text-slate-500 font-medium">
//             Powered by RAG + RL
//           </p>
//         </motion.div>
//       </motion.div>
//     </div>
//   )
// }

// export default WelcomeScreen


import { motion } from 'framer-motion'
import { BsRobot } from 'react-icons/bs'
import { HiSparkles, HiLightningBolt, HiDatabase } from 'react-icons/hi'
import { RiStarSFill } from 'react-icons/ri'
import Button from '../UI/Button'

const WelcomeScreen = ({ onNewChat }) => {
  const features = [
    {
      icon: <HiSparkles className="w-7 h-7" />,
      title: 'RAG-Powered',
      description: 'Retrieval-augmented generation for accurate banking information',
      color: 'from-cyan-400 to-blue-500'
    },
    {
      icon: <HiLightningBolt className="w-7 h-7" />,
      title: 'RL-Enhanced',
      description: 'Smart policy network decides when to fetch documents',
      color: 'from-purple-400 to-pink-500'
    },
    {
      icon: <HiDatabase className="w-7 h-7" />,
      title: 'Persistent Storage',
      description: 'Conversations stored in MongoDB Atlas cloud database',
      color: 'from-emerald-400 to-cyan-500'
    }
  ]

  const exampleQueries = [
    'What is my account balance?',
    'How do I open a savings account?',
    'What are the interest rates for fixed deposits?',
    'How can I apply for a credit card?'
  ]

  // Floating particles animation
  const particleVariants = {
    animate: (i) => ({
      y: [0, -30, 0],
      x: [0, Math.random() * 20 - 10, 0],
      opacity: [0.3, 0.8, 0.3],
      transition: {
        duration: 3 + i * 0.5,
        repeat: Infinity,
        ease: "easeInOut"
      }
    })
  }

  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen overflow-y-auto overflow-x-hidden bg-gradient-to-br from-slate-950 via-indigo-950 to-purple-950 dark:from-black dark:via-slate-950 dark:to-indigo-950">
      {/* Animated Background - Northern Lights Effect */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {/* Aurora Borealis Waves */}
        <motion.div
          className="absolute top-0 left-0 w-full h-full opacity-30"
          animate={{
            background: [
              'radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.3) 0%, transparent 50%)',
              'radial-gradient(circle at 80% 50%, rgba(139, 92, 246, 0.3) 0%, transparent 50%)',
              'radial-gradient(circle at 50% 80%, rgba(236, 72, 153, 0.3) 0%, transparent 50%)',
              'radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.3) 0%, transparent 50%)',
            ]
          }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
        />
        
        {/* Floating Light Particles */}
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            custom={i}
            variants={particleVariants}
            animate="animate"
            className="absolute w-1 h-1 bg-cyan-400 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              filter: 'blur(1px)',
            }}
          />
        ))}

        {/* Grid Overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(99,102,241,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(99,102,241,0.03)_1px,transparent_1px)] bg-[size:50px_50px]" />
      </div>

      {/* Main Content */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative z-10 max-w-6xl w-full px-4 sm:px-8 py-12 space-y-16"
      >
        {/* Logo & Title Section */}
        <div className="space-y-8 text-center">
          {/* Floating Robot with Glow */}
          <motion.div
            animate={{ 
              y: [0, -15, 0],
            }}
            transition={{ 
              duration: 4, 
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="inline-block relative"
          >
            {/* Multi-layer Glow */}
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 rounded-full blur-3xl opacity-40 animate-pulse" />
            <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-indigo-500 rounded-full blur-2xl opacity-30" />
            
            {/* Robot Icon */}
            <motion.div
              animate={{ 
                rotate: [0, 5, -5, 0],
              }}
              transition={{ 
                duration: 3, 
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <BsRobot className="relative w-32 h-32 text-transparent bg-gradient-to-br from-cyan-400 via-blue-500 to-purple-500 bg-clip-text drop-shadow-[0_0_15px_rgba(59,130,246,0.5)]" />
            </motion.div>

            {/* Orbiting Stars */}
            {[0, 120, 240].map((rotation, i) => (
              <motion.div
                key={i}
                className="absolute top-1/2 left-1/2"
                animate={{
                  rotate: 360,
                }}
                transition={{
                  duration: 8,
                  repeat: Infinity,
                  ease: "linear",
                  delay: i * 0.3
                }}
                style={{
                  transformOrigin: '0 0',
                }}
              >
                <RiStarSFill 
                  className="w-4 h-4 text-yellow-400"
                  style={{
                    transform: `translate(-50%, -50%) translateX(80px) rotate(${-rotation}deg)`,
                    filter: 'drop-shadow(0 0 4px rgba(250, 204, 21, 0.6))'
                  }}
                />
              </motion.div>
            ))}
          </motion.div>
          
          {/* Title with Neon Glow */}
          <div className="space-y-4">
            <motion.h1 
              className="text-6xl sm:text-7xl md:text-8xl font-black tracking-tight"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent drop-shadow-[0_0_30px_rgba(59,130,246,0.5)]">
                QuestRAG
              </span>
            </motion.h1>
            
            <motion.p 
              className="text-xl sm:text-2xl font-semibold text-transparent bg-gradient-to-r from-slate-300 via-cyan-200 to-blue-300 bg-clip-text"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              Your intelligent banking companion powered by AI
            </motion.p>

            {/* Glowing Underline */}
            <motion.div
              className="mx-auto w-32 h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent rounded-full"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 }}
            />
          </div>
        </div>

        {/* Feature Cards - Neon Style */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ 
                delay: index * 0.2 + 0.8,
                duration: 0.6,
                ease: "easeOut"
              }}
              whileHover={{ 
                y: -10,
                scale: 1.03,
                transition: { duration: 0.3 }
              }}
              className="group relative"
            >
              {/* Glowing Border Effect */}
              <div className={`absolute inset-0 bg-gradient-to-r ${feature.color} rounded-2xl opacity-0 group-hover:opacity-20 blur-xl transition-all duration-500`} />
              <div className={`absolute inset-0 bg-gradient-to-r ${feature.color} rounded-2xl opacity-10 blur-md`} />
              
              {/* Card */}
              <div className="relative h-full p-8 rounded-2xl bg-slate-900/50 backdrop-blur-xl border border-slate-800 group-hover:border-cyan-500/50 shadow-2xl group-hover:shadow-cyan-500/20 transition-all duration-500">
                <div className="flex flex-col items-center gap-5 text-center">
                  {/* Neon Icon */}
                  <motion.div 
                    whileHover={{ 
                      rotate: [0, -10, 10, 0],
                      scale: 1.15
                    }}
                    transition={{ duration: 0.5 }}
                    className={`p-5 rounded-xl bg-gradient-to-br ${feature.color} shadow-lg shadow-cyan-500/30 group-hover:shadow-cyan-500/60 transition-shadow duration-500`}
                  >
                    <div className="text-white">
                      {feature.icon}
                    </div>
                  </motion.div>
                  
                  {/* Title */}
                  <h3 className={`text-xl font-bold bg-gradient-to-r ${feature.color} bg-clip-text text-transparent`}>
                    {feature.title}
                  </h3>
                  
                  {/* Description */}
                  <p className="text-sm leading-relaxed text-slate-400 group-hover:text-slate-300 transition-colors duration-300">
                    {feature.description}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Example Queries - Futuristic Pills */}
        <div className="space-y-8">
          <motion.h3 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.4 }}
            className="text-2xl font-bold text-center bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent"
          >
            Try asking:
          </motion.h3>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {exampleQueries.map((query, index) => (
              <motion.button
                key={index}
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ 
                  delay: index * 0.15 + 1.6,
                  duration: 0.5,
                  ease: "easeOut"
                }}
                whileHover={{ 
                  scale: 1.05,
                  x: 8,
                }}
                whileTap={{ scale: 0.95 }}
                onClick={onNewChat}
                className="group relative p-5 text-left rounded-xl bg-slate-900/40 backdrop-blur-sm border border-slate-800 hover:border-cyan-500/50 shadow-lg hover:shadow-cyan-500/20 transition-all duration-300 overflow-hidden"
              >
                {/* Hover Glow */}
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/0 via-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                
                <div className="relative flex items-center gap-4">
                  {/* Pulsing Dot */}
                  <motion.div 
                    animate={{
                      scale: [1, 1.3, 1],
                      opacity: [0.5, 1, 0.5]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                    className="flex-shrink-0 w-2 h-2 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500"
                  />
                  
                  <span className="text-sm font-medium text-slate-300 group-hover:text-cyan-300 transition-colors duration-300">
                    {query}
                  </span>
                </div>
              </motion.button>
            ))}
          </div>
        </div>

        {/* CTA Button - Neon Glow */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 2, duration: 0.6 }}
          className="flex justify-center"
        >
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Button
              variant="primary"
              size="lg"
              onClick={onNewChat}
              className="relative px-12 py-5 text-lg font-bold overflow-hidden group"
            >
              {/* Animated Glow Background */}
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-600 opacity-75 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-purple-500 blur-xl opacity-50 group-hover:opacity-75 animate-pulse" />
              
              {/* Text */}
              <span className="relative z-10 flex items-center gap-3">
                <BsRobot className="w-6 h-6" />
                Start Chatting
                <motion.span
                  animate={{ x: [0, 5, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  →
                </motion.span>
              </span>
            </Button>
          </motion.div>
        </motion.div>

        {/* Footer Badge - Glowing */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2.2 }}
          className="text-center space-y-2"
        >
          <div className="inline-block px-6 py-2 rounded-full bg-slate-900/50 border border-cyan-500/30 backdrop-blur-sm">
            <p className="text-xs font-semibold text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text">
              Powered by RAG + RL
            </p>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default WelcomeScreen
