package com.finlogik.back.forum.controllers;


import com.finlogik.back.forum.entities.Comment;
import com.finlogik.back.forum.entities.CommentDTO;
import com.finlogik.back.forum.services.ICommentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/comment")
public class CommentController {

    @Autowired
    ICommentService commentService;

    @PostMapping("/addComment/{userId}/{idPost}")
    @ResponseBody
    public Comment addComment(@RequestBody Comment comment,
                        @PathVariable("userId") String userId,
                        @PathVariable("idPost") Long idPost){

        return commentService.addComment(comment, userId, idPost);
    }

    @PutMapping("/updateComment/{commentId}")
    @ResponseBody
    public Comment updateComment(@PathVariable Long commentId,
                           @RequestBody Comment comment) {
        comment.setIdComment(commentId);
        return commentService.updateComment(comment);
    }

    @DeleteMapping("/deleteComment/{commentId}")
    @ResponseBody
    public void deleteComment(@PathVariable("commentId") Long commentId) {

        commentService.deleteComment(commentId);
    }

    @GetMapping("/getAllComments")
    @ResponseBody
    public List<CommentDTO> getAllComments() {
        List<CommentDTO> listComments = commentService.getAllComments();
        return listComments;
    }

    @GetMapping("/getComment/{commentId}")
    @ResponseBody
    public Comment getComment(@PathVariable("commentId") Long commentId) {
        return commentService.getComment(commentId);
    }



    @GetMapping("/byPost/{postId}")
    @ResponseBody
    public List<CommentDTO> getCommentsByPostId(@PathVariable Long postId) {
        List<CommentDTO> listComments = commentService.getCommentsByPostId(postId);
        return listComments;
    }

}
