import { motion } from 'framer-motion';
import { BrainCircuit, Code2, Database, Terminal, Zap } from 'lucide-react';

const skillCategories = [
  {
    title: "AI Engineering",
    icon: <BrainCircuit className="w-5 h-5" />,
    skills: ["NLP", "LLMs", "Prompt Engineering", "Semantic Search", "Whisper", "OpenAI"],
    className: "md:col-span-2"
  },
  {
    title: "Full Stack",
    icon: <Code2 className="w-5 h-5" />,
    skills: ["React", "TypeScript", "Python", "Streamlit", "Node.js"],
    className: "md:col-span-1"
  },
  {
    title: "Databases",
    icon: <Database className="w-5 h-5" />,
    skills: ["PostgreSQL", "Supabase", "MySQL", "SQLite"],
    className: "md:col-span-1"
  },
  {
    title: "Analysis",
    icon: <Terminal className="w-5 h-5" />,
    skills: ["Pandas", "Scikit-learn", "NumPy", "Matplotlib"],
    className: "md:col-span-1"
  },
  {
    title: "Ecosystem",
    icon: <Zap className="w-5 h-5" />,
    skills: ["Git", "Docker", "REST APIs", "SEO"],
    className: "md:col-span-1"
  }
];

export const Skills = () => {
  return (
    <section className="py-32 bg-black px-4 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        <motion.div
           initial={{ opacity: 0, y: 20 }}
           whileInView={{ opacity: 1, y: 0 }}
           viewport={{ once: true }}
           className="mb-24"
        >
          <h2 className="text-4xl md:text-5xl font-medium text-white mb-6 tracking-tighter">Capabilities</h2>
          <p className="text-white/40 max-w-xl font-light">
            An intersection of artificial intelligence and scalable software engineering.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {skillCategories.map((cat, index) => (
            <motion.div
              key={cat.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              className={`p-10 rounded-sm border border-white/10 bg-white/[0.02] flex flex-col justify-between hover:bg-white/[0.04] transition-colors ${cat.className}`}
            >
              <div>
                <div className="text-white/20 mb-10">
                  {cat.icon}
                </div>
                <h3 className="text-lg font-medium text-white mb-6 tracking-tight">{cat.title}</h3>
              </div>

              <div className="flex flex-wrap gap-x-6 gap-y-3">
                {cat.skills.map(skill => (
                  <span key={skill} className="text-[11px] uppercase tracking-widest text-white/40 font-medium">
                    {skill}
                  </span>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
