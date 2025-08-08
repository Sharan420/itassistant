import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { ArrowUp } from "lucide-react";
import { ModeToggle } from "@/components/modeToggle";
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/appSidebar";

export default function ChatPage() {
  return (
    <>
      <AppSidebar />
      <SidebarInset>
        <header className='flex h-14 items-center gap-2 px-4'>
          <SidebarTrigger variant='outline' size='icon' />
          <div className='ml-auto'>
            <ModeToggle />
          </div>
        </header>

        <div className='relative flex-1'>
          <div
            aria-label='input chat'
            className='absolute bottom-0 left-1/2 -translate-x-1/2 px-2 pt-2 pb-0 bg-slate-200 dark:bg-slate-800/70 rounded-b-none rounded-t-xl max-w-4xl w-full'
          >
            <div className='flex flex-col gap-4 border-b-0 bg-slate-400/40 dark:bg-slate-800 rounded-b-none rounded-t-lg p-1 pb-2 px-2 items-end'>
              <Textarea
                placeholder='Type your message here...'
                className='p-2 resize-none w-full border-none focus-visible:ring-0 focus-visible:ring-offset-0 shadow-none dark:bg-transparent'
              />
              <Button variant='outline' size='icon'>
                <ArrowUp className='size-5' />
              </Button>
            </div>
          </div>
        </div>
      </SidebarInset>
    </>
  );
}
