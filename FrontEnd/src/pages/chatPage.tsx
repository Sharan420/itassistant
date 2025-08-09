"use client";
//Imports:
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useState } from "react";
import { cn } from "@/lib/utils";

import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { ArrowUp } from "lucide-react";
import { ModeToggle } from "@/components/modeToggle";
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/appSidebar";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";

//Typescript:
interface Message {
  id: number;
  text: string;
  sender: string;
}

const formSchema = z.object({
  message: z.string().min(1, { message: "Message is required" }),
});

//Components:
export default function ChatPage() {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      message: "",
    },
  });

  const [isRespPending, setIsRespPending] = useState(false);
  const [chat, setChat] = useState<Message[]>([
    {
      id: 1,
      text: "Welcome to the IT Assistant! How can I help you today?",
      sender: "assistant",
    },
  ]);

  async function handleSendMessage(values: z.infer<typeof formSchema>) {
    setChat((prevMessages) => [
      ...prevMessages,
      {
        id: prevMessages.length + 1,
        text: values.message,
        sender: "user",
      },
    ]);
    setIsRespPending(true);

    try {
      console.log(values.message);
      const response = await fetch("http://localhost:3000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: values.message }),
      });
      const data = await response.json();
      setChat((prevMessages) => [
        ...prevMessages,
        {
          id: prevMessages.length + 1,
          text: data.response,
          sender: "assistant",
        },
      ]);
      setIsRespPending(false);
    } catch (error) {
      console.error(error);
      setChat((prevMessages) => [
        ...prevMessages,
        {
          id: prevMessages.length + 1,
          text: "Error: " + error,
          sender: "assistant",
        },
      ]);
      setIsRespPending(false);
    }

    form.reset();
  }

  return (
    <>
      <AppSidebar />
      <SidebarInset>
        <div className='relative flex-1 flex justify-center items-start'>
          <header className='absolute top-0 left-0 w-full flex h-14 items-center gap-2 px-4'>
            <SidebarTrigger variant='outline' size='icon' />
            <div className='ml-auto'>
              <ModeToggle />
            </div>
          </header>
          <div className='flex flex-col gap-4 h-[calc(100vh-11rem)] w-full max-w-4xl rounded-lg rounded-t-none p-3 overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-slate-500 '>
            {chat.map((message) => (
              <div
                className={cn(
                  message.sender === "assistant"
                    ? "flex justify-start"
                    : "flex justify-end"
                )}
                key={message.id}
              >
                <div
                  className={cn(
                    "w-fit flex flex-col gap-4 border-b-0 rounded-md p-2 items-start",
                    message.sender === "assistant"
                      ? "bg-slate-400 dark:bg-slate-800"
                      : "bg-slate-200 dark:bg-slate-600"
                  )}
                >
                  <p>{message.text}</p>
                </div>
              </div>
            ))}
            {isRespPending && (
              <div className='w-fit bg-slate-400 dark:bg-slate-800 rounded-md animate-pulse p-2'>
                Loading ...
              </div>
            )}
          </div>
          <div
            aria-label='input chat'
            className='absolute bottom-0 left-1/2 -translate-x-1/2 px-2 pt-2 pb-0 bg-slate-200 dark:bg-slate-800/70 rounded-b-none rounded-t-xl max-w-4xl w-full'
          >
            <div className='flex flex-col gap-4 border-b-0 bg-slate-400/40 dark:bg-slate-800 rounded-b-none rounded-t-lg p-1 pb-2 px-2 items-end'>
              <Form {...form}>
                <form
                  onSubmit={form.handleSubmit(handleSendMessage)}
                  className='w-full flex flex-col items-end'
                >
                  <FormField
                    control={form.control}
                    name='message'
                    render={({ field }) => (
                      <FormItem className='w-full'>
                        <FormControl>
                          <Textarea
                            placeholder='Type your message here...'
                            className='p-2 resize-none w-full border-none focus-visible:ring-0 focus-visible:ring-offset-0 shadow-none dark:bg-transparent'
                            {...field}
                            value={field.value}
                            onChange={field.onChange}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <Button variant='outline' size='icon' type='submit'>
                    <ArrowUp className='size-5' />
                  </Button>
                </form>
              </Form>
            </div>
          </div>
        </div>
      </SidebarInset>
    </>
  );
}
