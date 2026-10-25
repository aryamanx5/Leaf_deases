import { ArrowUpRight } from 'lucide-react';
import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { DEPLOYED_APP_URL } from '@/content/brand';

/** "Run the model": the hosted analysis app when VITE_DEPLOYED_APP_URL is set, else this site's own /analyze page. */
export default function RunModelLink({ className, children }: { className: string; children: ReactNode }) {
  if (DEPLOYED_APP_URL) {
    return (
      <a href={DEPLOYED_APP_URL} target="_blank" rel="noreferrer" className={className}>
        {children}
        <ArrowUpRight size={16} aria-hidden="true" className="ml-1 inline" />
      </a>
    );
  }
  return (
    <Link to="/analyze" className={className}>
      {children}
    </Link>
  );
}
