import { Post } from "./Post";

export class Topic {
    idTopic: any;
    title: any;
    question: any;
    likes: any;
    dislikes: any;
    creationDate: Date;
    userId: any;
    user: any;
    posts: Post[];
  }

