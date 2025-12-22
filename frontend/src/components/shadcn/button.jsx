import * as React from "react";
import { cn } from "../../lib/utils";

const buttonVariants = {
  variant: {
    default: "bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 hover:from-violet-700 hover:via-purple-700 hover:to-fuchsia-700 text-white shadow-lg shadow-violet-500/30 hover:shadow-violet-500/50",
    destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
    outline: "border-2 border-violet-500 bg-transparent text-violet-600 dark:text-violet-400 hover:bg-violet-50 dark:hover:bg-violet-900/30",
    secondary: "bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 hover:bg-violet-200 dark:hover:bg-violet-900/50",
    ghost: "hover:bg-violet-100 dark:hover:bg-violet-900/30 text-violet-700 dark:text-violet-300",
    link: "text-violet-600 dark:text-violet-400 underline-offset-4 hover:underline",
  },
  size: {
    default: "h-10 px-4 py-2",
    sm: "h-9 rounded-md px-3",
    lg: "h-11 rounded-md px-8",
    icon: "h-10 w-10",
  },
};

export function getButtonClasses(variant = "default", size = "default") {
  return cn(
    "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
    buttonVariants.variant[variant],
    buttonVariants.size[size]
  );
}

const Button = React.forwardRef(
  ({ className, variant = "default", size = "default", asChild = false, children, ...props }, ref) => {
    return (
      <button
        className={cn(getButtonClasses(variant, size), className)}
        ref={ref}
        {...props}
      >
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";

export { Button, buttonVariants };
