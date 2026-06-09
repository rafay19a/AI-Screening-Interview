import { motion } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';

const projects = [
  {
    title: "Hired By AI",
    category: "AI Recruitment",
    impact: "60% reduction in HR workload",
    description: "An automated recruitment ecosystem using semantic search and voice-based AI interviewing. Built with Whisper, LLMs, and Streamlit.",
    link: "#",
    tags: ["NLP", "LLM", "Whisper", "Python"]
  },
  {
    title: "Buttertune",
    category: "E-commerce",
    impact: "Premium Skincare Platform",
    description: "A full-stack commerce engine designed for high-end organic products. Focused on SEO, speed, and conversion architecture.",
    link: "https://buttertuneskincare.com/",
    tags: ["Full Stack", "E-commerce", "SEO"]
  },
  {
    title: "Shama-e-Hayat",
    category: "E-commerce",
    impact: "International Expansion",
    description: "Digital storefront for premium organic candles, facilitating market expansion from Pakistan to France with localized experiences.",
    link: "https://shama-e-hayat.vercel.app",
    tags: ["Next.js", "Internationalization", "UX"]
  },
  {
    title: "Ora De Nuit",
    category: "Hospitality",
    impact: "Brand Digitization",
    description: "A sophisticated web presence for an American restaurant, focusing on visual storytelling and seamless reservation flows.",
    link: "https://ora-de-nuit.vercel.app",
    tags: ["React", "UI/UX", "Hospitality"]
  },
  {
    title: "Roameo PMS",
    category: "SaaS",
    impact: "Hospitality Scaling",
    description: "Property management system specifically engineered for resorts in Kashmir. Handles enterprise bookings and guest logic.",
    link: "https://roameo-pms.vercel.app/",
    tags: ["SaaS", "Management", "Property Tech"]
  },
  {
    title: "Kartarpur Journey",
    category: "Tourism",
    impact: "Heritage Tech",
    description: "Specialized platform for Sikh heritage tourism. Built with focus on accessibility and modern travel flows.",
    link: "https://kartarpur-journey-2uev.vercel.app",
    tags: ["Next.js", "UX", "Travel"]
  }
];

export const Projects = () => {
  return (
    <section id="projects" className="py-48 bg-black px-6 md:px-12 relative border-t border-white/5">
      <div className="max-w-screen-2xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-baseline mb-32 border-b border-white/10 pb-24">
          <h2 className="text-[8vw] md:text-[5vw] font-medium tracking-tighter-premium mb-8 md:mb-0">Selected<br />Works.</h2>
          <p className="text-white/40 max-w-sm font-light text-lg italic">
            "Engineering solutions that redefine how humans interact with technology."
          </p>
        </div>

        <div className="space-y-48">
          {projects.map((project) => (
            <motion.div
              key={project.title}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
              className="grid grid-cols-1 md:grid-cols-12 gap-12 group"
            >
              <div className="md:col-span-7 overflow-hidden bg-white/5 aspect-[16/9] relative">
                <div className="absolute inset-0 flex items-center justify-center opacity-20 group-hover:scale-105 transition-transform duration-1000">
                  <span className="text-[10vw] font-bold tracking-tighter uppercase opacity-10">{project.title.slice(0, 2)}</span>
                </div>
                <a href={project.link} target="_blank" rel="noreferrer" className="absolute inset-0 z-10 cursor-none" />
              </div>

              <div className="md:col-span-5 flex flex-col justify-between py-4">
                <div>
                  <div className="flex justify-between items-baseline mb-12">
                    <span className="text-[10px] uppercase tracking-[0.3em] text-white/40">{project.category}</span>
                    <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-white/80">{project.impact}</span>
                  </div>

                  <h3 className="text-5xl font-medium tracking-tighter mb-8 group-hover:translate-x-2 transition-transform duration-500">{project.title}</h3>
                  <p className="text-white/50 text-xl font-light leading-relaxed mb-12">
                    {project.description}
                  </p>
                </div>

                <div>
                  <div className="flex flex-wrap gap-x-8 gap-y-4 mb-12">
                    {project.tags.map(tag => (
                        <span key={tag} className="text-[10px] uppercase tracking-widest text-white/30">{tag}</span>
                    ))}
                  </div>
                  <a href={project.link} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.3em] font-bold group-hover:gap-4 transition-all">
                    View Project <ArrowUpRight className="w-4 h-4" />
                  </a>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
