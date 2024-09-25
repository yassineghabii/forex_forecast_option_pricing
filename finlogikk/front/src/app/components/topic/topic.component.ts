import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Topic } from 'src/app/entities/Topic';
import { KeycloakService } from "../../services/keycloak/keycloak.service";
import { TopicService } from "../../services/services/topic.service";
import { faThumbsUp, faThumbsDown, faTrashAlt, faEdit, faComment, faCommentAlt, faCommentDots, faComments} from '@fortawesome/free-regular-svg-icons'
import { faReply,faBan ,faListAlt, faThumbsUp as faThumbsUpp, faThumbsDown as faThumbsDownn } from '@fortawesome/free-solid-svg-icons'
import { Router  } from '@angular/router';
import { WindowRef } from "../../services/services/window-ref.service";
import * as bootstrap from 'bootstrap';
import { NgbModal } from '@ng-bootstrap/ng-bootstrap';



@Component({
  selector: 'app-topic',
  templateUrl: './topic.component.html',
  styleUrls: ['./topic.component.scss']
})
export class TopicComponent implements OnInit {

  faThumbsUp = faThumbsUp;
  faThumbsUpp = faThumbsUpp;
  faThumbsDown = faThumbsDown;
  faThumbsDownn = faThumbsDownn;
  faTrashAlt = faTrashAlt;
  faEdit = faEdit;
  faComment = faComment;
  faCommentAlt = faCommentAlt;
  faCommentDots = faCommentDots;
  faComments = faComments;
  faReply = faReply;
  faListAlt = faListAlt;
  faBan = faBan;


  listTopic:any;
  Topic:Topic;
  TopicDetails = null as any;
  Topicupdate:any;
  closeResult = '';

  topicForm: FormGroup;

  currentPage = 1;
  pageSize = 10;

  Topict: any = {
  };

  idUser: any;
  users: any;

  userVoteStatus: string;


  constructor(private formBuilder: FormBuilder,
              private TopicService: TopicService,
              private keycloakService: KeycloakService,
              private router: Router,
              private windowRef: WindowRef,
              private modalService: NgbModal
  ) { }

  ngOnInit(): void {



    this.keycloakService.getAllUsers().subscribe(data => {
      this.users = data;
    });


    this.topicForm = this.formBuilder.group({
      title: [''],
      question: [''],
      user: ['']
    });

    this.getAllTopic()
    this.Topic = {
      idTopic: null,
      title: null,
      question: null,
      likes: null,
      dislikes: null,
      creationDate: null,
      userId: null,
      user: null,
      posts: null
    }
    this.Topicupdate = {
      idTopic: null,
      title: null,
      question: null,
      likes: null,
      dislikes: null,
      creationDate: null,
      user: null,
      posts: null
    }

  }



  viewTopicDetails(idTopic: number) {
    this.router.navigate(['/topic', idTopic]);
  }

//   onSubmit() {
//     if (this.topicForm.invalid) {
//       return;
//     }

//     const formData = new FormData();
//     formData.append('idTopic', this.topicForm.controls['title'].value);
//     formData.append('title', this.topicForm.controls['title'].value);
//     formData.append('user', this.topicForm.controls['title'].value);
//     formData.append('posts', this.topicForm.controls['title'].value);

//   }


  isCurrentUserOwner(topic: Topic): boolean {
    const userId = this.getUserId();
    return userId !== null && topic.userId === userId;
  }


  getAllTopic() {
    this.TopicService.getAllTopics().subscribe(res =>{
      this.listTopic= res
      console.log('Topics:', res);

      this.listTopic.forEach((topic: any) => {

        this.TopicService.postCount(topic.idTopic).subscribe((count: number) => {
          topic.postCount = count;
        });

        this.TopicService.getUserVoteStatus(topic.idTopic, this.getUserId()).subscribe((userVoteStatus: string) => {
          topic.userVoteStatus = userVoteStatus;
        });

      });


    });
  }

  addTopic1(){
    this.TopicService.addTopic(this.Topic, this.idUser).subscribe(()=> this.getAllTopic());
  }

  addTopic() {
    if (this.topicForm.invalid) {
      console.log('Invalid form');
      return;
    }

    const userId = this.getUserId();

    this.TopicService.addTopic(this.topicForm.value, userId).subscribe(
      (resp) => {
        console.log('Topic added successfully:', resp);
        this.getAllTopic();
        //this.topicForm.reset();
        // window.location.reload();
      },
      (err) => {
        console.log('Error while adding topic:', err);
      }
    );
  }


  deleteTopic(idTopic:number){
    this.TopicService.deleteTopic(idTopic).subscribe(()=>this.getAllTopic() ,res=>{
      this.listTopic=res;
      this.getAllTopic();
      // window.location.reload();
    });

  }

  getTopicdetails(){
    this.TopicService.getTopic(this.Topicupdate.idTopic).subscribe(res=>{this.TopicDetails=res});
  }

  edit(topic: any){
    this.Topicupdate = topic;
  }

  add(topic: any){
    this.Topic = topic;
  }




  votelike(idTopic: any) {

    const userId = this.getUserId();

    this.TopicService.voteLike(idTopic, userId).subscribe(
      (resp) => {
        console.log('Topic liked successfully:', resp);
        this.getAllTopic();
      },
      (err) => {
        console.error('Error while liking topic:', err);
      }
    );
  }

  votedislike(idTopic: any) {

    const userId = this.getUserId();

    this.TopicService.voteDislike(idTopic, userId).subscribe(
      (resp) => {
        console.log('Topic disliked successfully:', resp);
        this.getAllTopic();
      },
      (err) => {
        console.error('Error while disliking topic:',idTopic , err);
      }
    );
  }






  confirmerSuppression(idTopic: number) {
    const estConfirme = this.windowRef.nativeWindow.confirm("Are you sure you want to delete this topic?");
    if (estConfirme) {
      this.deleteTopic(idTopic);
    }
  }

  // openModal() {
  //   const modalElement = document.getElementById('exampleModal1');
  //   const modal = new bootstrap.Modal(modalElement);
  //   modal.show();
  // }

  openModal(content: any) {
    this.modalService.open(content, { ariaLabelledBy: 'modal-basic-title' });
  }


  getUserId(): string | null {
    const tokenParsed = this.keycloakService.keycloak.tokenParsed; // Keycloak décode automatiquement le token JWT
    const userId = tokenParsed ? tokenParsed.sub : null; // 'sub' est l'ID utilisateur
    return userId;
  }
}
