import { motion, type Variants } from 'framer-motion';
import { ExternalLink, ShoppingCart, Plane, Utensils, Hotel, Sparkles } from 'lucide-react';

const projects = [
  {
    title: "Buttertune",
    description: "Full-stack e-commerce platform for organic skincare. High-performance catalog and secure checkout integration.",
    link: "https://buttertuneskincare.com/",
    tags: ["E-commerce", "Full Stack"],
    icon: <ShoppingCart className="w-5 h-5" />
  },
  {
    title: "Roameo PMS",
    description: "Enterprise-grade hotel management system. Automated bookings and property scaling solutions.",
    link: "https://roameo-pms.vercel.app/",
    tags: ["SaaS", "Management"],
    icon: <Hotel className="w-5 h-5" />
  },
  {
    title: "Kartarpur Journey",
    description: "Sikh tourism platform facilitating heritage travel and seamless user experiences.",
    link: "https://kartarpur-journey-2uev.vercel.app",
    tags: ["Next.js", "Hospitality"],
    icon: <Plane className="w-5 h-5" />
  },
  {
    title: "Ora De Nuit",
    description: "Digital presence for high-end American dining, focusing on minimalist UI and speed.",
    link: "https://ora-de-nuit.vercel.app/",
    tags: ["UI/UX", "Restaurant"],
    icon: <Utensils className="w-5 h-5" />
  },
  {
    title: "Shama-e-Hayat",
    description: "Market expansion platform for organic products, bridging Pakistan and France.",
    link: "https://shama-e-hayat.vercel.app/",
    tags: ["Global", "E-commerce"],
    icon: <Sparkles className="w-5 h-5" />
  }
];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.3
    }
  }
};

const item: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] } }
};

export const Projects = () => {
  return (
    <section id="projects" className="py-32 bg-black px-4 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-24"
        >
          <h2 className="text-4xl md:text-5xl font-medium text-white mb-6 tracking-tighter">Selected Projects</h2>
          <p className="text-white/40 max-w-xl font-light">
            A curated selection of digital products built with a focus on clean code and exceptional user experience.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-24"
        >
          {projects.map((project) => (
            <motion.div
              key={project.title}
              variants={item}
              className="group"
            >
              <a href={project.link} target="_blank" rel="noreferrer" className="block">
                <div className="aspect-[16/10] bg-white/5 border border-white/10 rounded-sm mb-8 overflow-hidden relative flex items-center justify-center grayscale hover:grayscale-0 transition-all duration-700">
                  <div className="text-white/20 group-hover:text-white/80 transition-colors duration-500 scale-150 group-hover:scale-125 transition-transform duration-700">
                    {project.icon}
                  </div>
                  <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  <div className="absolute top-6 right-6 p-2 bg-black/80 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                    <ExternalLink className="w-4 h-4 text-white" />
                  </div>
                </div>

                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-xl font-medium text-white mb-2 tracking-tight group-hover:pl-2 transition-all duration-300">
                      {project.title}
                    </h3>
                    <p className="text-sm text-white/40 font-light mb-4 max-w-sm leading-relaxed">
                      {project.description}
                    </p>
                    <div className="flex gap-4">
                      {project.tags.map(tag => (
                        <span key={tag} className="text-[10px] uppercase tracking-widest text-white/30 font-medium">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </a>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};
